#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>


#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <Sacado.hpp>

#include "Tet4.hpp"
#include "Thermal.hpp"
#include "Solutions.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "WorksetBase.hpp"
#include "ComputedField.hpp"
#include "ScalarProduct.hpp"
#include "ProjectToNode.hpp"
#include "ThermalContent.hpp"
#include "GradientMatrix.hpp"
#include "alg/ParallelComm.hpp"
#include "ThermalMassMaterial.hpp"
#include "InterpolateFromNodal.hpp"
#include "alg/CrsLinearProblem.hpp"
#include "GeneralFluxDivergence.hpp"
#include "parabolic/VectorFunction.hpp"
#include "parabolic/PhysicsScalarFunction.hpp"

#include <fenv.h>

TEUCHOS_UNIT_TEST( HeatEquationTests, 3D )
{ 
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = typename Plato::ThermalElement<Plato::Tet4>;

  int numCells = tMesh->NumElements();
  constexpr int spaceDim     = ElementType::mNumSpatialDims;
  constexpr int nodesPerCell = ElementType::mNumNodesPerCell;
  constexpr int dofsPerCell  = ElementType::mNumDofsPerCell;

  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> T_host( tMesh->NumNodes() );
  Plato::Scalar Tval = 0.0, dval = 1.0;
  for( auto& val : T_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    T_host_view(T_host.data(),T_host.size());
  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);

  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto numPoints = tCubWeights.size();

  Plato::WorksetBase<ElementType> worksetBase(tMesh);

  Plato::ScalarArray3DT<Plato::Scalar>
    gradient("gradient",numCells,nodesPerCell,spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    tCellGrad("temperature gradient", numCells, spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    tCellFlux("thermal flux", numCells, spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    result("result", numCells, dofsPerCell);

  Plato::ScalarMultiVectorT<Plato::Scalar> tCellGpTemperature("cell temperature at step k", numCells, numPoints);

  Plato::ScalarMultiVectorT<Plato::Scalar> tCellGpThermalContent("cell heat content at step k", numCells, numPoints);

  Plato::ScalarMultiVectorT<Plato::Scalar> 
    massResult("mass", numCells, dofsPerCell);

  Plato::ScalarArray3DT<Plato::Scalar>
    configWS("config workset",numCells, nodesPerCell, spaceDim);
  worksetBase.worksetConfig(configWS);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    stateWS("state workset",numCells, dofsPerCell);
  worksetBase.worksetState(T, stateWS);

  Plato::ComputeGradientMatrix<ElementType> computeGradient;

  Plato::ScalarGrad<ElementType> scalarGrad;

  Plato::InterpolateFromNodal<ElementType, ElementType::mNumDofsPerNode> tInterpolateFromNodal;

  Plato::ThermalMassModelFactory<spaceDim> mmmfactory(*tParamList);
  auto thermalMassMaterialModel = mmmfactory.create("Unobtainium");

  Plato::ThermalContent<spaceDim> computeThermalContent(thermalMassMaterialModel);
  Plato::ProjectToNode<ElementType> projectThermalContent;

  Plato::ThermalConductionModelFactory<spaceDim> mmfactory(*tParamList);
  auto tMaterialModel = mmfactory.create("Unobtainium");

  Plato::ThermalFlux<ElementType>           thermalFlux(tMaterialModel);
  Plato::GeneralFluxDivergence<ElementType> fluxDivergence;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);


  Kokkos::parallel_for("flux divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {numCells, numPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Scalar tVolume(0.0);

    Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, Plato::Scalar> tGradient;

    Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tGrad(0.0);
    Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tFlux(0.0);

    auto tCubPoint = tCubPoints(iGpOrdinal);
    auto tBasisValues = ElementType::basisValues(tCubPoint);

    computeGradient(iCellOrdinal, tCubPoint, configWS, tGradient, tVolume);
    tVolume *= tCubWeights(iGpOrdinal);

    Kokkos::atomic_add(&cellVolume(iCellOrdinal), tVolume);

    for(Plato::OrdinalType iNode=0; iNode<nodesPerCell; iNode++)
    {
      for(Plato::OrdinalType iDim=0; iDim<spaceDim; iDim++)
      {
        gradient(iCellOrdinal, iNode, iDim) = tGradient(iNode, iDim);
      }
    }

    scalarGrad(iCellOrdinal, tGrad, stateWS, tGradient);
    for(Plato::OrdinalType iDim=0; iDim<spaceDim; iDim++)
    {
      tCellGrad(iCellOrdinal, iDim) = tGrad(iDim);
    }

    Plato::Scalar tTemperature(0.0);
    tInterpolateFromNodal(iCellOrdinal, tBasisValues, stateWS, tTemperature);
    tCellGpTemperature(iCellOrdinal, iGpOrdinal) = tTemperature;
    thermalFlux(tFlux, tGrad, tTemperature);
    for(Plato::OrdinalType iDim=0; iDim<spaceDim; iDim++)
    {
      tCellFlux(iCellOrdinal, iDim) = tFlux(iDim);
    }

    fluxDivergence(iCellOrdinal, result, tFlux, tGradient, tVolume);

    Plato::Scalar tThermalContent(0.0);
    computeThermalContent(tThermalContent, tTemperature, tTemperature);
    tCellGpThermalContent(iCellOrdinal, iGpOrdinal) = tThermalContent;

    projectThermalContent(iCellOrdinal, tVolume, tBasisValues, tThermalContent, massResult);
  });
  
  // test cell volume
  //
  auto cellVolume_Host = Kokkos::create_mirror_view( cellVolume );
  Kokkos::deep_copy( cellVolume_Host, cellVolume );

  std::vector<Plato::Scalar> cellVolume_gold = { 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333
  };

  int numGoldCells=cellVolume_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(cellVolume_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(cellVolume_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(cellVolume_Host(iCell), cellVolume_gold[iCell], 1e-13);
    }
  }

  // test state values
  //
  auto tTemperature_Host = Kokkos::create_mirror_view( tCellGpTemperature );
  Kokkos::deep_copy( tTemperature_Host, tCellGpTemperature );

  std::vector<Plato::Scalar> tTemperature_gold = { 
    10.23606797749978, 6.211145618000156, 10.68328157299974, 4.869504831500283, 
    5.105572809000069, 5.552786404500027, 9.577708763999649, 3.763932022500195, 
    5.276393202250006, 3.934752415750130, 9.301315561749627, 3.487538820250173, 
    4.763932022500198, 8.788854381999819, 10.13049516849969, 4.316718427000239, 
    9.894427190999906, 9.447213595499949, 11.23606797749978, 5.422291236000327, 
    9.723606797749971, 11.06524758424984, 11.51246117974980, 5.698684438250349, 
    11.23606797749978, 7.211145618000156, 11.68328157299974, 5.869504831500284, 
    6.105572809000070, 6.552786404500026, 10.57770876399965, 4.763932022500195, 
    6.276393202250005, 4.934752415750131, 10.30131556174963, 4.487538820250173, 
    5.763932022500198, 9.788854381999819, 11.13049516849969, 5.316718427000239, 
    10.89442719099991, 10.44721359549995, 12.23606797749978, 6.422291236000327
  };

  int tIndex=0;
  numGoldCells=tTemperature_gold.size()/4;
  for(int iCell=0; iCell<numGoldCells; iCell++)
  {
      for(int jGp=0; jGp<numPoints; ++jGp)
      {
          if(tTemperature_gold[tIndex] == 0.0)
          {
              TEST_ASSERT(fabs(tTemperature_Host(iCell, jGp)) < 1e-12);
          } 
          else 
          {
              TEST_FLOATING_EQUALITY(tTemperature_Host(iCell, jGp), tTemperature_gold[tIndex], 1e-13);
          }
          tIndex++;
      }
  }

  // test thermal content
  //
  auto thermalContent_Host = Kokkos::create_mirror_view( tCellGpThermalContent );
  Kokkos::deep_copy( thermalContent_Host, tCellGpThermalContent );

  std::vector<Plato::Scalar> thermalContent_gold = { 
    3.070820393249934e6, 1.863343685400047e6, 3.204984471899921e6, 1.460851449450085e6,
    1.531671842700021e6, 1.665835921350008e6, 2.873312629199895e6, 1.129179606750058e6, 
    1.582917960675002e6, 1.180425724725039e6, 2.790394668524888e6, 1.046261646075052e6, 
    1.429179606750059e6, 2.636656314599946e6, 3.039148550549908e6, 1.295015528100072e6, 
    2.968328157299972e6, 2.834164078649985e6, 3.370820393249934e6, 1.626687370800098e6, 
    2.917082039324991e6, 3.319574275274953e6, 3.453738353924941e6, 1.709605331475105e6, 
    3.370820393249934e6, 2.163343685400047e6, 3.504984471899921e6, 1.760851449450085e6, 
    1.831671842700021e6, 1.965835921350008e6, 3.173312629199895e6, 1.429179606750058e6, 
    1.882917960675001e6, 1.480425724725039e6, 3.090394668524887e6, 1.346261646075052e6, 
    1.729179606750059e6, 2.936656314599946e6, 3.339148550549908e6, 1.595015528100072e6, 
    3.268328157299972e6, 3.134164078649984e6, 3.670820393249934e6, 1.926687370800098e6, 
    3.217082039324991e6, 3.619574275274953e6, 3.753738353924941e6, 2.009605331475105e6
  };

  tIndex=0;
  numGoldCells=thermalContent_gold.size()/4;
  for(int iCell=0; iCell<numGoldCells; iCell++)
  {
      for(int jGp=0; jGp<numPoints; ++jGp)
      {
          if(thermalContent_gold[tIndex] == 0.0)
          {
              TEST_ASSERT(fabs(thermalContent_Host(iCell, jGp)) < 1e-12);
          } 
          else 
          {
              TEST_FLOATING_EQUALITY(thermalContent_Host(iCell, jGp), thermalContent_gold[tIndex], 1e-13);
          }
          tIndex++;
      }
  }

  // test gradient operator
  //
  auto gradient_Host = Kokkos::create_mirror_view( gradient );
  Kokkos::deep_copy( gradient_Host, gradient );

  std::vector<std::vector<std::vector<Plato::Scalar>>> gradient_gold = { 
    {{ 0.0,-2.0, 0.0}, { 2.0, 0.0,-2.0}, {-2.0, 2.0, 0.0}, {0.0, 0.0, 2.0}},
    {{ 0.0,-2.0, 0.0}, { 0.0, 2.0,-2.0}, {-2.0, 0.0, 2.0}, {2.0, 0.0, 0.0}},
    {{ 0.0, 0.0,-2.0}, {-2.0, 2.0, 0.0}, { 0.0,-2.0, 2.0}, {2.0, 0.0, 0.0}},
    {{ 0.0, 0.0,-2.0}, {-2.0, 0.0, 2.0}, { 2.0,-2.0, 0.0}, {0.0, 2.0, 0.0}},
    {{-2.0, 0.0, 0.0}, { 0.0,-2.0, 2.0}, { 2.0, 0.0,-2.0}, {0.0, 2.0, 0.0}},
    {{-2.0, 0.0, 0.0}, { 2.0,-2.0, 0.0}, { 0.0, 2.0,-2.0}, {0.0, 0.0, 2.0}}
  };

  numGoldCells=gradient_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iNode=0; iNode<nodesPerCell; iNode++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        if(gradient_gold[iCell][iNode][iDim] == 0.0){
          TEST_ASSERT(fabs(gradient_Host(iCell,iNode,iDim)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(gradient_Host(iCell,iNode,iDim), gradient_gold[iCell][iNode][iDim], 1e-13);
        }
      }
    }
  }

  // test temperature gradient
  //
  auto tgrad_Host = Kokkos::create_mirror_view( tCellGrad );
  Kokkos::deep_copy( tgrad_Host, tCellGrad );

  std::vector<std::vector<Plato::Scalar>> tgrad_gold = { 
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 },
    { 18.000, 6.000, 2.000 }
  };

  for(int iCell=0; iCell<int(tgrad_gold.size()); iCell++){
    for(int iDim=0; iDim<spaceDim; iDim++){
      if(tgrad_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tgrad_Host(iCell,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tgrad_Host(iCell,iDim), tgrad_gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test thermal flux
  //
  auto tflux_Host = Kokkos::create_mirror_view( tCellFlux );
  Kokkos::deep_copy( tflux_Host, tCellFlux );

  std::vector<std::vector<Plato::Scalar>> tflux_gold = { 
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 },
   {-18.0e6, -6.0e6, -2.0e6 }
  };

  for(int iCell=0; iCell<int(tflux_gold.size()); iCell++){
    for(int iDim=0; iDim<spaceDim; iDim++){
      if(tflux_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tflux_Host(iCell,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tflux_Host(iCell,iDim), tflux_gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test flux divergence
  //
  auto result_Host = Kokkos::create_mirror_view( result );
  Kokkos::deep_copy( result_Host, result );

  std::vector<std::vector<Plato::Scalar>> result_gold = { 
   { 250000.0000000000, -666666.6666666666, 500000.0000000000, -83333.33333333333 },
   { 250000.0000000000, -166666.6666666667, 666666.6666666666, -750000.0000000000 },
   { 83333.33333333333,  500000.0000000000, 166666.6666666667, -750000.0000000000 }
  };

  for(int iCell=0; iCell<int(result_gold.size()); iCell++){
    for(int iDof=0; iDof<dofsPerCell; iDof++){
      if(result_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(result_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(result_Host(iCell,iDof), result_gold[iCell][iDof], 1e-13);
      }
    }
  }

  // test projected thermal content
  //
  auto mass_result_Host = Kokkos::create_mirror_view( massResult );
  Kokkos::deep_copy( mass_result_Host, massResult );

  std::vector<std::vector<Plato::Scalar>> mass_result_gold = { 

   {10312.50000000001, 14062.50000000001, 11250.00000000001, 14374.99999999991},
   {7812.499999999994, 8749.999999999993, 9062.499999999995, 11874.99999999992},
   {7187.499999999991, 8437.499999999991, 7499.999999999991, 11249.99999999992},
   {9062.500000000000, 9375.000000000000, 12187.50000000000, 13124.99999999991}
  };

    for(int iCell=0; iCell<int(mass_result_gold.size()); iCell++)
    {
        for(int iNode=0; iNode<nodesPerCell; iNode++)
        {
            if(mass_result_gold[iCell][iNode] == 0.0)
            {
                TEST_ASSERT(fabs(mass_result_Host(iCell,iNode)) < 1e-12);
            } 
            else 
            {
                TEST_FLOATING_EQUALITY(mass_result_Host(iCell,iNode), mass_result_gold[iCell][iNode], 1e-13);
            }
        }
    }
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, HeatEquationResidual3D )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Parabolic'/>        \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                \n"
    "  <ParameterList name='Parabolic'>                                          \n"
    "    <ParameterList name='Penalty Function'>                                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e3'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Time Integration'>                                   \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>              \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                 \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> T_host( tMesh->NumNodes() );
  Plato::Scalar Tval = 0.0, dval = 1.0000;
  for( auto& val : T_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    T_host_view(T_host.data(),T_host.size());
  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);

  std::vector<Plato::Scalar> Tdot_host( tMesh->NumNodes() );
  Tval = 0.0; dval = 0.5000;
  for( auto& val : Tdot_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    Tdot_host_view(Tdot_host.data(),Tdot_host.size());
  auto Tdot = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), Tdot_host_view);


  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  Plato::Parabolic::VectorFunction<::Plato::Thermal<Plato::Tet4>>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));


  // compute and test value
  //
  auto timeStep = tParamList->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  auto residual = vectorFunction.value(T, Tdot, z, timeStep);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
    -2166637.6041666669771075, -2999957.8125000000000000, -833323.3333333332557231,
    -2499946.8750000000000000, -4499918.7500000009313226, -1999975.0000000000000000,
    -333318.6458333332557231, -1499970.3124999997671694, -1166651.6666666667442769,
    -999914.0625000002328306, -1499871.8750000004656613, -499960.9375000000582077,
    -499856.2499999998835847, 262.5000000004656613, 500118.7500000003492460,
    500048.4374999998253770, 1500134.3749999997671694, 1000089.0624999997671694,
    1166695.4166666665114462, 1500057.8124999997671694, 333362.3958333332557231,
    2000062.5000000000000000, 4500181.2500000000000000, 2500121.8750000004656613,
    833367.0833333331393078, 3000132.8125000000000000, 2166768.8541666669771075,
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-10);
    }
  }

  // compute and test gradient wrt state, T. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(T, Tdot, z, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
   499999.999999999942, -166666.666666666657, -166666.666666666657, 0,
  -166666.666666666657, 0, 0, 0, -166666.666666666657,
   833333.333333333372, -166666.666666666657, -250000.000000000000, 0,
  -250000.000000000000, 0, 0, 0, -166666.666666666657,
   333333.333333333314, -83333.3333333333285, -83333.3333333333285, 0
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt previous state, T. (i.e., jacobian)
  //
  auto jacobian_v = vectorFunction.gradient_v(T, Tdot, z, timeStep);

  auto jac_v_entries = jacobian_v->entries();
  auto jac_v_entriesHost = Kokkos::create_mirror_view( jac_v_entries );
  Kokkos::deep_copy(jac_v_entriesHost, jac_v_entries);

  std::vector<Plato::Scalar> gold_jac_v_entries = {
    3.75000000000000977, 0.625000000000002887, 0.625000000000002887, 
    0.625000000000002887, 0.625000000000002887, 0.625000000000002887, 
    0.625000000000002887, 1.87499999999999067, 0.625000000000002887, 
    5.00000000000001332, 0.625000000000002887, 0.937500000000004441, 
    0.625000000000002887, 0.937500000000004441, 0.625000000000002887,
    1.24999999999999978, 1.87499999999999067, 0.625000000000002887, 
    1.25000000000000311, 0.312500000000001499, 0.312500000000001499, 
    0.624999999999996891
  };

  int jac_v_entriesSize = gold_jac_v_entries.size();
  for(int i=0; i<jac_v_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_v_entriesHost(i), gold_jac_v_entries[i], 1.0e-14);
  }


  // compute and test objective gradient wrt control, z
  //
  auto gradient_z = vectorFunction.gradient_z(T, Tdot, z, timeStep);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
    -541657.0136455872561783, -208331.7904794917849358, -83331.5345112734939903,
    -291665.1073170731542632, 291669.2333933811169118, 83335.6605875814420870,
    208335.9165557996893767, 541673.6979166636010632, -41665.1238128263357794,
    -749985.5267691994085908, -208331.4779794917558320, -208330.4490091494517401,
    -291664.7948170731542632, 416670.6180501623894088, 83335.9730875814420870,
    458338.3333333326736465, 541674.6354166636010632, -41664.8113128263430553,
    -208329.7631236119777896, -124999.0707478759577498, 125001.2284067813016009,
    250002.4167775329842698
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto gradient_x = vectorFunction.gradient_x(T, Tdot, z, timeStep);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
    -9000023.7500000000000000, -3000018.1250000000000000, -1000016.2499999998835847,
    2833333.3333333330228925, 833333.3333333332557231, 2333316.4583333334885538,
    2500000.0000000000000000, 2666646.6666666669771075, -166666.6666666666569654,
    -666666.6666666666278616, 1499990.9375000002328306, 1499991.5625000000000000,
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-10);
  }

}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, InternalThermalEnergy3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Parabolic'/>        \n"
    "  <Parameter name='Objective' type='string' value='My Internal Thermal Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                 \n"
    "  <ParameterList name='Criteria'>                                          \n"
    "    <ParameterList name='Internal Energy'>                                 \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>       \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermal Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                              \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>             \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>        \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                \n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e3'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Time Integration'>                                   \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>              \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                 \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  // create mesh based temperature from host data
  //
  int tNumSteps = 3;
  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarMultiVector T("temperature history", tNumSteps, tNumNodes);
  Plato::ScalarMultiVector Tdot("temperature rate history", tNumSteps, tNumNodes);
  Plato::ScalarVector z("density", tNumNodes);
  Kokkos::parallel_for("temperature history", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     for( int i=0; i<tNumSteps; i++){
       T(i, aNodeOrdinal) = (i+1)*aNodeOrdinal;
       Tdot(i, aNodeOrdinal) = 0.0;
     }
  });


  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);

  std::string tMyFunction("Internal Energy");
  Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<Plato::Tet4>>
    scalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);

  auto timeStep = tParamList->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  int timeIncIndex = 1;

  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", T);
  tSolution.set("StateDot", Tdot);
  auto value = scalarFunction.value(tSolution, z, timeStep);

  Plato::Scalar value_gold = 4.73200000000000095e9;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(tSolution, z, timeIncIndex, timeStep);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -8.666666666666668e6, -1.200000000000000e7,
  -3.333333333333333e6, -1.000000000000000e7,
  -1.800000000000000e7, -7.999999999999999e6,
  -1.333333333333333e6, -6.000000000000001e6,
  -4.666666666666665e6, -3.999999999999999e6,
  -6.000000000000002e6, -1.999999999999998e6
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = scalarFunction.gradient_z(tSolution, z, timeStep);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   1.478750000000000e8, 1.971666666666667e8,
   4.929166666666666e7, 1.971666666666667e8,
   2.957500000000000e8, 9.858333333333334e7,
   4.929166666666667e7, 9.858333333333333e7,
   4.929166666666666e7, 1.971666666666667e8,
   2.957500000000000e8, 9.858333333333334e7,
   2.957500000000000e8, 5.914999999999999e8,
   2.957500000000000e8, 9.858333333333333e7,
   2.957500000000000e8, 1.971666666666667e8,
   4.929166666666667e7, 9.858333333333333e7,
   4.929166666666666e7, 9.858333333333334e7,
   2.957499999999999e8, 1.971666666666667e8,
   4.929166666666666e7, 1.971666666666666e8,
   1.478750000000000e8
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z, timeStep);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   6.196666666666666e8, -5.633333333333331e7,
  -2.816666666666667e8,  8.125000000000000e8,
  -1.235000000000000e8,  1.560000000000000e8,
   1.928333333333333e8, -6.716666666666669e7,
   4.376666666666666e8,  5.785000000000000e8,
   3.900000000000000e8, -4.615000000000000e8,
   9.230000000000000e8,  7.020000000000001e8,
   2.340000000000000e8,  3.444999999999999e8,
   3.119999999999999e8,  6.955000000000000e8,
  -4.116666666666667e7,  4.463333333333334e8,
  -1.798333333333333e8,
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}


/******************************************************************************/
/*! 
  \brief Create a 'ComputedField' object for a uniform scalar field
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, ComputedField_UniformScalar )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  // compute fields
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                \n"
    "  <ParameterList name='Computed Fields'>                            \n"
    "    <ParameterList name='Uniform Initial Temperature'>              \n"
    "      <Parameter name='Function' type='string' value='100.0'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear X Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*x'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear Y Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*y'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear Z Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*z'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Bilinear XY Initial Temperature'>          \n"
    "      <Parameter name='Function' type='string' value='1.0*x*y'/>    \n"
    "    </ParameterList>                                                \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                    \n"
  );

  auto tComputedFields = Plato::ComputedFields<spaceDim>(tMesh, tParamList->sublist("Computed Fields"));

  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector T("temperature", tNumNodes);

  tComputedFields.get("Uniform Initial Temperature", T);

  // pull temperature to host
  //
  auto T_Host = Kokkos::create_mirror_view( T );
  Kokkos::deep_copy( T_Host, T );

  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 100.0, 1e-15);
  }


  Plato::ScalarVector xcoords("x", tNumNodes);
  Plato::ScalarVector ycoords("y", tNumNodes);
  Plato::ScalarVector zcoords("z", tNumNodes);
  auto coords = tMesh->Coordinates();
  Kokkos::parallel_for("get coords", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(int nodeOrdinal)
  {
    xcoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+0];
    ycoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+1];
    zcoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+2];
  });

  auto xCoords_Host = Kokkos::create_mirror_view( xcoords );
  auto yCoords_Host = Kokkos::create_mirror_view( ycoords );
  auto zCoords_Host = Kokkos::create_mirror_view( zcoords );
  Kokkos::deep_copy( xCoords_Host, xcoords );
  Kokkos::deep_copy( yCoords_Host, ycoords );
  Kokkos::deep_copy( zCoords_Host, zcoords );

  tComputedFields.get("Linear X Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*xCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Linear Y Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*yCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Linear Z Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*zCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Bilinear XY Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*xCoords_Host[iNode]*yCoords_Host[iNode], 1e-15);
  }
}
