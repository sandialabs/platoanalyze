#include "util/PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <Sacado.hpp>

#include "Tet4.hpp"
#include "BLAS1.hpp"
#include "WorksetBase.hpp"
#include "StateValues.hpp"
#include "SpatialModel.hpp"
#include "ComputedField.hpp"
#include "ProjectToNode.hpp"
#include "GradientMatrix.hpp"
#include "ThermalContent.hpp"
#include "PressureDivergence.hpp"
#include "InterpolateFromNodal.hpp"
#include "GeneralFluxDivergence.hpp"
#include "GeneralStressDivergence.hpp"

#include "parabolic/AbstractVectorFunction.hpp"

#include "stabilized/Projection.hpp"
#include "stabilized/TMKinetics.hpp"
#include "stabilized/TMKinematics.hpp"
#include "stabilized/VectorFunction.hpp"
#include "stabilized/Thermomechanics.hpp"
#include "stabilized/ThermomechanicsElement.hpp"

TEUCHOS_UNIT_TEST( StabilizedThermomechTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = Plato::Stabilized::ThermomechanicsElement<Plato::Tet4>;

  int tNumCells = tMesh->NumElements();
  constexpr int spaceDim      = ElementType::mNumSpatialDims;
  constexpr int numVoigtTerms = ElementType::mNumVoigtTerms;
  constexpr int nodesPerCell  = ElementType::mNumNodesPerCell;
  constexpr int dofsPerCell   = ElementType::mNumDofsPerCell;
  constexpr int dofsPerNode   = ElementType::mNumDofsPerNode;

  static constexpr int PDofOffset = spaceDim;
  static constexpr int TDofOffset = spaceDim+1;

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+2); // displacements + pressure + temperature
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  Plato::ScalarMultiVector tPGradWS("Projected pressure gradient workset", tNumDofs, spaceDim*nodesPerCell);
  Kokkos::parallel_for("projected pgrad", Kokkos::RangePolicy<int>(0,tNumCells), KOKKOS_LAMBDA(const int & aCellOrdinal)
  {
      for(int iNode=0; iNode<nodesPerCell; iNode++)
      {
          for(int iDim=0; iDim<spaceDim; iDim++)
          {
              tPGradWS(aCellOrdinal, iNode*spaceDim+iDim) = (4e-7)*(iNode+1)*(iDim+1)*(aCellOrdinal+1);
          }
      }
  });

  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (5e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+4) = (4e-7)*aNodeOrdinal;
  });

  Plato::WorksetBase<ElementType> worksetBase(tMesh);

  Plato::ScalarArray3D     configWS           ("config workset",     tNumCells, nodesPerCell, spaceDim);
  Plato::ScalarMultiVector tStressDivResult   ("stress div",         tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tPressureDivResult ("pressure div",       tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tStabDivResult     ("stabilization div",  tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tFluxDivResult     ("thermal flux div",   tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tVolResult         ("volume diff proj",   tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tMassResult        ("mass",               tNumCells, dofsPerCell);
  Plato::ScalarMultiVector tStateWS           ("state workset",      tNumCells, dofsPerCell);

  Plato::ScalarVector      tCellVolume            ("cell volume",        tNumCells);

  worksetBase.worksetConfig(configWS);

  worksetBase.worksetState(state, tStateWS);

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                      \n"
    "  <ParameterList name='Material Models'>                                                  \n"
    "    <ParameterList name='Unobtainium'>                                                    \n"
    "      <ParameterList name='Thermal Mass'>                                                 \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      </ParameterList>                                                                    \n"
    "      <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>   \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "</ParameterList>                                                                          \n"
  );

  Plato::ThermalMassModelFactory<spaceDim> mmmfactory(*params);
  auto massMaterialModel = mmmfactory.create("Unobtainium");

  Plato::LinearThermoelasticModelFactory<spaceDim> mmfactory(*params);
  auto materialModel = mmfactory.create("Unobtainium");

  Plato::ComputeGradientMatrix<ElementType>    computeGradient;
  Plato::Stabilized::TMKinematics<ElementType> kinematics;
  Plato::Stabilized::TMKinetics<ElementType>   kinetics(materialModel);

  Plato::InterpolateFromNodal<ElementType, spaceDim, 0, spaceDim> interpolatePGradFromNodal;
  Plato::InterpolateFromNodal<ElementType, dofsPerNode, PDofOffset> interpolatePressureFromNodal;
  Plato::InterpolateFromNodal<ElementType, dofsPerNode, TDofOffset> interpolateTemperatureFromNodal;

  Plato::GeneralFluxDivergence   <ElementType, dofsPerNode, TDofOffset> fluxDivergence;
  Plato::GeneralFluxDivergence   <ElementType, dofsPerNode, PDofOffset> stabDivergence;
  Plato::GeneralStressDivergence <ElementType, dofsPerNode>             stressDivergence;

  Plato::PressureDivergence <ElementType, dofsPerNode> pressureDivergence;

  Plato::ThermalContent<spaceDim> computeThermalContent(massMaterialModel);

  Plato::ProjectToNode<ElementType, dofsPerNode, PDofOffset> projectVolumeStrain;
  Plato::ProjectToNode<ElementType, dofsPerNode, TDofOffset> projectThermalContent;

  Plato::Scalar tTimeStep = 2.0;


  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();

  Plato::ScalarArray3D tCellStab           ("cell,gp stabilization",     tNumCells, tNumPoints, spaceDim);
  Plato::ScalarArray3D tCellTFlux          ("cell,gp thermal flux",      tNumCells, tNumPoints, spaceDim);
  Plato::ScalarArray3D tCellProjectedPGrad ("cell,gp projected p grad",  tNumCells, tNumPoints, spaceDim);
  Plato::ScalarArray3D tCellPressureGrad   ("cell,gp pressure grad",     tNumCells, tNumPoints, spaceDim);
  Plato::ScalarArray3D tCellTGrad          ("cell,gp Temperature grad",  tNumCells, tNumPoints, spaceDim);
  Plato::ScalarArray3D tCellDevStress      ("cell,gp deviatoric stress", tNumCells, tNumPoints, numVoigtTerms);
  
  Plato::ScalarMultiVector tCellThermalContent ("cell,gp heat at step k", tNumCells, tNumPoints);
  Plato::ScalarMultiVector tCellTemperature    ("cell,gp temperature",    tNumCells, tNumPoints);
  Plato::ScalarMultiVector tCellVolStrain      ("cell,gp volume strain",  tNumCells, tNumPoints);

  Kokkos::parallel_for("compute residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
  {
    Plato::Scalar tVolume(0.0);

    Plato::Matrix<ElementType::mNumNodesPerCell, spaceDim> tGradient;

    // compute gradient operator and cell volume
    //
    auto tCubPoint = tCubPoints(iGpOrdinal);
    computeGradient(iCellOrdinal, tCubPoint, configWS, tGradient, tVolume);
    tVolume *= tCubWeights(iGpOrdinal);

    // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
    //
    Plato::Array<numVoigtTerms> tDGrad(0.0);
    Plato::Array<spaceDim> tPGrad(0.0);
    Plato::Array<spaceDim> tTGrad(0.0);
    kinematics(iCellOrdinal, tDGrad, tPGrad, tTGrad, tStateWS, tGradient);

    auto tBasisValues = ElementType::basisValues(tCubPoint);
    Plato::Array<spaceDim> tProjectedPGrad(0.0);
    interpolatePGradFromNodal(iCellOrdinal, tBasisValues, tPGradWS, tProjectedPGrad);

    Plato::Scalar tPressure(0.0);
    interpolatePressureFromNodal(iCellOrdinal, tBasisValues, tStateWS, tPressure);

    Plato::Scalar tTemperature(0.0);
    interpolateTemperatureFromNodal(iCellOrdinal, tBasisValues, tStateWS, tTemperature);

    // compute the constitutive response
    //
    Plato::Scalar tVolStrain(0.0);
    Plato::Array<spaceDim> tGPStab(0.0);
    Plato::Array<spaceDim> tTFlux(0.0);
    Plato::Array<numVoigtTerms> tDevStress(0.0);
    kinetics(tVolume, tProjectedPGrad, tDGrad, tPGrad, tTGrad, tTemperature,
             tPressure, tDevStress, tVolStrain, tTFlux, tGPStab);

    tCellTemperature(iCellOrdinal, iGpOrdinal) = tTemperature;
    tCellVolStrain(iCellOrdinal, iGpOrdinal) = tVolStrain;

    Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
    for(Plato::OrdinalType iVoigt=0; iVoigt<numVoigtTerms; iVoigt++)
    {
      tCellDevStress(iCellOrdinal, iGpOrdinal, iVoigt) = tDevStress(iVoigt);
    }
    for(Plato::OrdinalType iDim=0; iDim<spaceDim; iDim++)
    {
      tCellStab(iCellOrdinal, iGpOrdinal, iDim) = tGPStab(iDim);
      tCellTFlux(iCellOrdinal, iGpOrdinal, iDim) = tTFlux(iDim);
      tCellProjectedPGrad(iCellOrdinal, iGpOrdinal, iDim) = tProjectedPGrad(iDim);
      tCellPressureGrad(iCellOrdinal, iGpOrdinal, iDim) = tPGrad(iDim);
      tCellTGrad(iCellOrdinal, iGpOrdinal, iDim) = tTGrad(iDim);
    }

    stressDivergence   (iCellOrdinal, tStressDivResult,   tDevStress, tGradient, tVolume, tTimeStep/2.0);
    pressureDivergence (iCellOrdinal, tPressureDivResult, tPressure,  tGradient, tVolume, tTimeStep/2.0);
    stabDivergence     (iCellOrdinal, tStabDivResult,     tGPStab,    tGradient, tVolume, tTimeStep/2.0);
    fluxDivergence     (iCellOrdinal, tFluxDivResult,     tTFlux,     tGradient, tVolume, tTimeStep/2.0);

    Plato::Scalar tThermalContent(0.0);
    computeThermalContent(tThermalContent, tTemperature);
    tCellThermalContent(iCellOrdinal,iGpOrdinal) = tThermalContent;

    projectVolumeStrain  (iCellOrdinal, tVolume, tBasisValues, tVolStrain, tVolResult);
    projectThermalContent(iCellOrdinal, tVolume, tBasisValues, tThermalContent, tMassResult);

  });

  {
    // test gp temperatures
    //
    auto tTemperature_Host = Kokkos::create_mirror_view( tCellTemperature );
    Kokkos::deep_copy( tTemperature_Host, tCellTemperature );

    std::vector<std::vector<Plato::Scalar>> tGold = {
      {3.69442719099992102e-6, 2.08445824720007253e-6, 3.87331262919990465e-6, 1.54780193260012310e-6},
      {1.64222912360003772e-6, 1.82111456180002072e-6, 3.43108350559986942e-6, 1.10557280900008808e-6}
    };

    int tNumCells=tGold.size();
    int tNumGPs=tGold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGP=0; iGP<tNumGPs; iGP++){
        if(tGold[iCell][iGP] == 0.0){
          TEST_ASSERT(fabs(tTemperature_Host(iCell,iGP)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tTemperature_Host(iCell,iGP), tGold[iCell][iGP], 1e-13);
        }
      }
    }
  }

  {
    // test deviatoric stress
    //
    auto tDevStress_Host = Kokkos::create_mirror_view( tCellDevStress );
    Kokkos::deep_copy( tDevStress_Host, tCellDevStress );

    std::vector<std::vector<std::vector<double>>> tGold = {{
      { 40026.6844563111663,0.00000000000000000000,-40026.6844562962651,73382.2548365577095,186791.194129419659,140093.395597064722}
    }};

    int tNumCells=tGold.size();
    int tNumGp=tGold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGp=0; iGp<tNumGp; iGp++){
        for(int iVoigt=0; iVoigt<numVoigtTerms; iVoigt++){
          if(tGold[iCell][iGp][iVoigt] == 0.0){
            TEST_ASSERT(fabs(tDevStress_Host(iCell,iGp,iVoigt)) < 1e-8);
          } else {
            TEST_FLOATING_EQUALITY(tDevStress_Host(iCell, iGp, iVoigt), tGold[iCell][iGp][iVoigt], 1e-12);
          }
        }
      }
    }
  }

  {
    // test volume strain
    //
    auto tVolStrain_Host = Kokkos::create_mirror_view( tCellVolStrain );
    Kokkos::deep_copy( tVolStrain_Host, tCellVolStrain );

    std::vector<std::vector<Plato::Scalar>> tGold = {
      {3.59988916718423414e-6, 3.59993746625254817e-6, 3.59988380062097228e-6, 3.59995356594198646e-6},
      {3.59995073312625635e-6, 3.59994536656311011e-6, 3.59989706749468004e-6, 3.59996683281569422e-6}
    };

    int tNumCells=tGold.size();
    int tNumGPs=tGold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGP=0; iGP<tNumGPs; iGP++){
        TEST_FLOATING_EQUALITY(tVolStrain_Host(iCell,iGP), tGold[iCell][iGP], 1e-13);
      }
    }
  }

  {
    // test cell stabilization
    //
    auto tCellStab_Host = Kokkos::create_mirror_view( tCellStab );
    Kokkos::deep_copy( tCellStab_Host, tCellStab );

    std::vector<std::vector<std::vector<Plato::Scalar>>> tGold = 
    {{{3.64350540274726886e-18,  5.30972974585768664e-19, -7.79949365333393180e-19},
      {3.56293495022208300e-18,  3.69832069535398151e-19, -1.02166072290894902e-18},
      {3.48236449769689791e-18,  2.08691164485027808e-19, -1.26337208048450467e-18},
      {3.72407585527245394e-18,  6.92113879636139224e-19, -5.38238007757837339e-19}},
     {{3.23338810694927503e-18, -2.89261617010216049e-19, -2.01030125272737003e-18},
      {3.07224720189890486e-18, -6.11543427110957218e-19, -2.49372396787848133e-18},
      {2.91110629684853391e-18, -9.33825237211697857e-19, -2.97714668302959301e-18},
      {3.39452901199964598e-18,  3.30201930905249090e-20, -1.52687853757625835e-18}}};

    int tNumCells=tGold.size();
    int tNumGP=tGold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGp=0; iGp<tNumGP; iGp++){
        for(int iDim=0; iDim<spaceDim; iDim++){
          TEST_FLOATING_EQUALITY(tCellStab_Host(iCell, iGp, iDim), tGold[iCell][iGp][iDim], 1e-13);
        }
      }
    }
  }

  {
    // test thermal flux
    //
    auto tflux_Host = Kokkos::create_mirror_view( tCellTFlux );
    Kokkos::deep_copy( tflux_Host, tCellTFlux );

    std::vector<std::vector<std::vector<Plato::Scalar>>> tflux_gold = 
    {{{0.00719999999999999980,0.00239999999999999979,0.000800000000000000797},
      {0.00719999999999999980,0.00239999999999999979,0.000800000000000000797},
      {0.00719999999999999980,0.00239999999999999979,0.000800000000000000797},
      {0.00719999999999999980,0.00239999999999999979,0.000800000000000000797}},
     {{0.00720000000000000067,0.00239999999999999979,0.000799999999999999930},
      {0.00720000000000000067,0.00239999999999999979,0.000799999999999999930},
      {0.00720000000000000067,0.00239999999999999979,0.000799999999999999930},
      {0.00720000000000000067,0.00239999999999999979,0.000799999999999999930}}};

    int tNumCells=tflux_gold.size();
    int tNumGp=tflux_gold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGp=0; iGp<tNumGp; iGp++){
        for(int iDim=0; iDim<spaceDim; iDim++){
          if(tflux_gold[iCell][iGp][iDim] == 0.0){
            TEST_ASSERT(fabs(tflux_Host(iCell,iGp,iDim)) < 1e-12);
          } else {
            TEST_FLOATING_EQUALITY(tflux_Host(iCell,iGp,iDim), tflux_gold[iCell][iGp][iDim], 1e-13);
          }
        }
      }
    }
  }

  {
    // test cell volume
    //
    auto tCellVolume_Host = Kokkos::create_mirror_view( tCellVolume );
    Kokkos::deep_copy( tCellVolume_Host, tCellVolume );

    std::vector<Plato::Scalar> gold = { 
      0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
      0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
      0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
      0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333
    };

    int numCells=gold.size();
    for(int iCell=0; iCell<numCells; iCell++){
      if(gold[iCell] == 0.0){
        TEST_ASSERT(fabs(tCellVolume_Host(iCell)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tCellVolume_Host(iCell), gold[iCell], 1e-13);
      }
    }
  }


  {
    // test thermal content
    //
    auto tThermalContent_Host = Kokkos::create_mirror_view( tCellThermalContent );
    Kokkos::deep_copy( tThermalContent_Host, tCellThermalContent );

    std::vector<std::vector<Plato::Scalar>> gold = 
      {{1.10832815729997636,  0.625337474160021722, 1.16199378875997139,  0.464340579780037011},
       {0.492668737080011299, 0.546334368540006166, 1.02932505167996080,  0.331671842700026420},
       {0.513167184270003629, 0.352170289890018751, 0.996157867409958153, 0.298504658430023773},
       {0.451671842700026693, 0.934662525839981106, 1.09565942021996610,  0.398006211240031715},
       {1.06733126291999181,  1.01366563145999677,  1.22832815729997669,  0.530674948320042250},
       {1.04683281572999953,  1.20782971010998441,  1.26149534156997944,  0.563842132590044898},
       {1.22832815729997646,  0.745337474160021829, 1.28199378875997150,  0.584340579780036951},
       {0.612668737080011239, 0.666334368540006161, 1.14932505167996091,  0.451671842700026416}};

    int tNumCells=gold.size();
    int tNumGPs=gold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGp=0; iGp<tNumGPs; iGp++){
        if(gold[iCell][iGp] == 0.0){
          TEST_ASSERT(fabs(tThermalContent_Host(iCell,iGp)) < 1e-9);
        } else {
          TEST_FLOATING_EQUALITY(tThermalContent_Host(iCell,iGp), gold[iCell][iGp], 1e-9);
        }
      }
    }
  }

  {
    // test interpolated pressure gradient
    //
    auto tProjectedPGrad_Host = Kokkos::create_mirror_view( tCellProjectedPGrad );
    Kokkos::deep_copy( tProjectedPGrad_Host, tCellProjectedPGrad );

    std::vector<std::vector<std::vector<Plato::Scalar>>> gold =
    {{{9.10557280900009619e-7, 1.82111456180001924e-6, 2.73167184270002843e-6},
      {1.08944271909999283e-6, 2.17888543819998566e-6, 3.26832815729997849e-6},
      {1.26832815729997583e-6, 2.53665631459995166e-6, 3.80498447189992771e-6},
      {7.31671842700026407e-7, 1.46334368540005281e-6, 2.19501552810007922e-6}},
     {{1.82111456180001924e-6, 3.64222912360003848e-6, 5.46334368540005687e-6},
      {2.17888543819998566e-6, 4.35777087639997133e-6, 6.53665631459995699e-6},
      {2.53665631459995166e-6, 5.07331262919990333e-6, 7.60996894379985542e-6},
      {1.46334368540005281e-6, 2.92668737080010563e-6, 4.39003105620015844e-6}}};

    int tNumCells = gold.size();
    int tNumGps = gold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGp=0; iGp<tNumGps; iGp++){
        for(int iDim=0; iDim<spaceDim; iDim++){
          TEST_FLOATING_EQUALITY(tProjectedPGrad_Host(iCell,iGp,iDim), gold[iCell][iGp][iDim], 1e-13);
        }
      }
    }
  }

  {
    // test pressure gradient
    //
    auto tPressureGrad_Host = Kokkos::create_mirror_view( tCellPressureGrad );
    Kokkos::deep_copy( tPressureGrad_Host, tCellPressureGrad );

    std::vector<std::vector<std::vector<Plato::Scalar>>> gold =
    {{{9.00000000000000023e-6, 3.00000000000000008e-6, 9.9999999999999890e-7 },
      {9.00000000000000023e-6, 3.00000000000000008e-6, 9.9999999999999890e-7 },
      {9.00000000000000023e-6, 3.00000000000000008e-6, 9.9999999999999890e-7 },
      {9.00000000000000023e-6, 3.00000000000000008e-6, 9.9999999999999890e-7 }},
     {{8.99999999999999853e-6, 3.00000000000000008e-6, 9.99999999999999743e-7},
      {8.99999999999999853e-6, 3.00000000000000008e-6, 9.99999999999999743e-7},
      {8.99999999999999853e-6, 3.00000000000000008e-6, 9.99999999999999743e-7},
      {8.99999999999999853e-6, 3.00000000000000008e-6, 9.99999999999999743e-7}}};

    int tNumCells = gold.size();
    int tNumGps = gold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGp=0; iGp<tNumGps; iGp++){
        for(int iDim=0; iDim<spaceDim; iDim++){
          TEST_FLOATING_EQUALITY(tPressureGrad_Host(iCell,iGp,iDim), gold[iCell][iGp][iDim], 1e-13);
        }
      }
    }
  }

  {
    // test temperature gradient
    //
    auto tgrad_Host = Kokkos::create_mirror_view( tCellTGrad );
    Kokkos::deep_copy( tgrad_Host, tCellTGrad );

    std::vector<std::vector<std::vector<Plato::Scalar>>> gold =
      {{{7.19999999999999967e-6, 2.39999999999999989e-6, 8.00000000000000811e-7},
        {7.19999999999999967e-6, 2.39999999999999989e-6, 8.00000000000000811e-7},
        {7.19999999999999967e-6, 2.39999999999999989e-6, 8.00000000000000811e-7},
        {7.19999999999999967e-6, 2.39999999999999989e-6, 8.00000000000000811e-7}},
       {{7.20000000000000052e-6, 2.39999999999999989e-6, 7.99999999999999964e-7},
        {7.20000000000000052e-6, 2.39999999999999989e-6, 7.99999999999999964e-7},
        {7.20000000000000052e-6, 2.39999999999999989e-6, 7.99999999999999964e-7},
        {7.20000000000000052e-6, 2.39999999999999989e-6, 7.99999999999999964e-7}}};

    int tNumCells = gold.size();
    int tNumGps = gold[0].size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iGp=0; iGp<tNumGps; iGp++){
        for(int iDim=0; iDim<spaceDim; iDim++){
          if(gold[iCell][iGp][iDim] == 0.0){
            TEST_ASSERT(fabs(tgrad_Host(iCell,iGp,iDim)) < 1e-12);
          } else {
            TEST_FLOATING_EQUALITY(tgrad_Host(iCell,iGp,iDim), gold[iCell][iGp][iDim], 1e-13);
          }
        }
      }
    }
  }

  {
    // test stress divergence and local assembly
    //
    auto tStressDivResult_Host = Kokkos::create_mirror_view( tStressDivResult );
    Kokkos::deep_copy( tStressDivResult_Host, tStressDivResult );

    std::vector<std::vector<double>> gold =
    {{-5837.22481654436342, 0.00000000000000000, -3057.59395152323759, 0.000000000000000000, 0.000000000000000000,
      -6115.18790304649701, 2779.63086502112537,  9450.74494107180726, 0.000000000000000000, 0.000000000000000000,
       4169.44629753170830,-5837.22481654444073, -4725.37247053591454, 0.000000000000000000, 0.000000000000000000,
       7782.96642205915214, 3057.59395152323759, -1667.77851901265467, 0.000000000000000000, 0.000000000000000000}};

    int tNumCells = gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tStressDivResult_Host(iCell,iDof)) < 1e-8);
        } else {
          TEST_FLOATING_EQUALITY(tStressDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-12);
        }
      }
    }
  }

  {
    // test pressure divergence and local assembly
    //
    auto tPressureDivResult_Host = Kokkos::create_mirror_view( tPressureDivResult );
    Kokkos::deep_copy( tPressureDivResult_Host, tPressureDivResult );

    std::vector<std::vector<Plato::Scalar>> gold = 
    {{0.000000000000000000,  -1.45833333333333613e-7, 0.000000000000000000,   0.000000000000000000, 0.000000000000000000, 
      1.45833333333333613e-7, 0.000000000000000000,  -1.45833333333333613e-7, 0.000000000000000000, 0.000000000000000000,
     -1.45833333333333613e-7, 1.45833333333333613e-7, 0.000000000000000000,   0.000000000000000000, 0.000000000000000000,
      0.000000000000000000,   0.000000000000000000,   1.45833333333333613e-7, 0.000000000000000000, 0.000000000000000000}};

    int tNumCells = gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tPressureDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tPressureDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  {
    // test stabilization divergence and local assembly
    //
    auto tStabDivResult_Host = Kokkos::create_mirror_view( tStabDivResult );
    Kokkos::deep_copy( tStabDivResult_Host, tStabDivResult );

    std::vector<std::vector<Plato::Scalar>> gold =
    {{0.000000000000000000, 0.000000000000000000, 0.000000000000000000, -1.87667717525243086e-20, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000,  1.87667717525243646e-19, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, -1.31367402267670519e-19, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, -3.75335435050487918e-20, 0.000000000000000000}};

    int tNumCells = gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tStabDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tStabDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  {
    // test thermal flux divergence and local assembly
    //
    auto tFluxDivResult_Host = Kokkos::create_mirror_view( tFluxDivResult );
    Kokkos::deep_copy( tFluxDivResult_Host, tFluxDivResult );

    std::vector<std::vector<Plato::Scalar>> gold =
    {{0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, -0.0000999999999999999912,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000,  0.000266666666666666625,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, -0.000200000000000000010,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000,  0.0000333333333333333620}};

    int tNumCells = gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tFluxDivResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tFluxDivResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  {
    // test projected volume eqn
    //
    auto tVolResult_Host = Kokkos::create_mirror_view( tVolResult );
    Kokkos::deep_copy( tVolResult_Host, tVolResult );

    std::vector<std::vector<Plato::Scalar>> gold =
    {{0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.87496499999996229e-8, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.87494999999997670e-8, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.87496124999997648e-8, 0.000000000000000000,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 1.87494874999994966e-8, 0.000000000000000000}};

    int tNumCells = gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tVolResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tVolResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }

  {
    // test projected mass
    //
    auto tMassResult_Host = Kokkos::create_mirror_view( tMassResult );
    Kokkos::deep_copy( tMassResult_Host, tMassResult );

    std::vector<std::vector<Plato::Scalar>> gold =
    {{0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00349999999999998316,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00500000000000001658,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00387500000000001645,
      0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00512500000000001669}};

    int tNumCells = gold.size();
    for(int iCell=0; iCell<tNumCells; iCell++){
      for(int iDof=0; iDof<dofsPerCell; iDof++){
        if(gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tMassResult_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tMassResult_Host(iCell,iDof), gold[iCell][iDof], 1e-13);
        }
      }
    }
  }
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         StabilizedThermomechResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( StabilizedThermomechTests, StabilizedThermomechResidual3D )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = Plato::Stabilized::ThermomechanicsElement<Plato::Tet4>;

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+2);
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  Plato::ScalarVector tState        ("state",         tNumDofs);
  Plato::ScalarVector tControl      ("control",       tNumNodes);
  Plato::ScalarVector tProjPGrad    ("ProjPGrad",     tNumNodes*spaceDim);
  Plato::ScalarVector tProjectState ("Project state", tNumNodes);

  Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
  {
     tControl(aNodeOrdinal) = 1.0;

     tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+4) = (4e-7)*aNodeOrdinal;
     tState(aNodeOrdinal*tNumDofsPerNode+3) =    0.0*aNodeOrdinal;
  });


  // create input for stabilized thermomechanics
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>                       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                              \n"
    "  <ParameterList name='Stabilized Elliptic'>                                              \n"
    "    <ParameterList name='Penalty Function'>                                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                              \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                                 \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Spatial Model'>                                                    \n"
    "    <ParameterList name='Domains'>                                                        \n"
    "      <ParameterList name='Design Volume'>                                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                      \n"
    "        <Parameter name='Material Model' type='string' value='Kryptonite'/>               \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Material Models'>                                                  \n"
    "    <ParameterList name='Kryptonite'>                                                     \n"
    "      <ParameterList name='Thermal Mass'>                                                 \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      </ParameterList>                                                                    \n"
    "      <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>   \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Time Stepping'>                                                    \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>                            \n"
    "    <Parameter name='Time Step' type='double' value='1.0'/>                               \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Newton Iteration'>                                                 \n"
    "    <Parameter name='Number Iterations' type='int' value='2'/>                            \n"
    "  </ParameterList>                                                                        \n"
    "</ParameterList>                                                                          \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
  Plato::Stabilized::VectorFunction<::Plato::Stabilized::Thermomechanics<Plato::Tet4>>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // create input pressure gradient projector
  //
  Teuchos::RCP<Teuchos::ParameterList> paramsProjector =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "      <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "      <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>   \n"
    "      <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

  // copy projection state
  Plato::blas1::extract<Plato::Stabilized::ThermomechanicsElement<Plato::Tet4>::mNumDofsPerNode,
                 Plato::Stabilized::Thermomechanics<Plato::Tet4>::ProjectorType::ElementType::mProjectionDof>(tState, tProjectState);


  // create constraint evaluator
  //
  Plato::Stabilized::VectorFunction<::Plato::Stabilized::Projection<Plato::Tet4, ElementType::mNumDofsPerNode, ElementType::mPressureDofOffset>>
    tProjectorVectorFunction(tSpatialModel, tDataMap, *paramsProjector, "State Gradient Projection");

  auto tProjResidual = tProjectorVectorFunction.value      (tProjPGrad, tProjectState, tControl);
  auto tProjJacobian = tProjectorVectorFunction.gradient_u (tProjPGrad, tProjectState, tControl);


  Plato::Solve::RowSummed<spaceDim>(tProjJacobian, tProjPGrad, tProjResidual);


  // compute and test value
  //
  auto timeStep = params->sublist("Time Stepping").get<Plato::Scalar>("Time Step");
  auto tResidual = vectorFunction.value(tState, tProjPGrad, tControl, timeStep);

  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<double> tGold =
  {
    -30575.9395155763268, -17789.6375363854313, -18345.5637093583755, 1.12498049999996111e-7, -0.000866666666666666627,
    -22515.0100071816269, -17511.6744500184250, -32521.6811207887404, 1.49997124999994593e-7, -0.00119999999999999946,
     8060.92950841537277,  277.963086387605472, -14176.1174113472043, 3.74993249999989460e-8, -0.000333333333333333376,
    -28352.2348238301493, -26684.4563043277230, -18345.5637095877137, 1.49996249999993140e-7, -0.00100000000000000002,
    -10006.6711151388827, -35023.3488994538275, -46697.7985324173933, 2.24994249999989958e-7, -0.00180000000000000017, 
     18345.5637086913339, -8338.89259512603167, -28352.2348227046859, 7.49982499999972219e-8, -0.000800000000000000147,
  };

  int tNumCells = tGold.size();
  for(int iNode=0; iNode<tNumCells; iNode++){
    if(tGold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidual_Host[iNode]) < 1e-11);
    } else {
      TEST_FLOATING_EQUALITY(tResidual_Host[iNode], tGold[iNode], 1e-9);
    }
  }


  // compute and test gradient wrt state. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(tState, tProjPGrad, tControl, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries =
{1.85308724334749870e10, 0, 0, -0.0208333333333332107, 0, 0, 
 1.85308724334745827e10, 0, -0.0208333333333332107, 0, 0, 0, 
 1.85308724334745789e10, -0.0208333333333332107, 0,
-0.0208333333333332107, -0.0208333333333332107,
-0.0208333333333332107, -2.25701261030292215e-13,
-3.74999999999997151e-7, 0, 0, 0, 0, 499.999999999999943,
-5.55926173004224968e9, 0, 
 2.77963086502112484e9, 0, 0, 0, -5.55926173004224968e9, 
 2.77963086502112484e9, 0, 0, -1.85308724334731412e9,
-1.85308724334731412e9, -7.41234897339007950e9,
-0.0208333333333333738, 0, -0.0104166666666666054,
-0.0104166666666666054, 0.0208333333333332107, 
 7.49837536767640727e-14, -6.24999999999997060e-8, 0, 0, 0, 0,
-166.666666666666657, -5.55926173004224968e9, 
 2.77963086502112484e9, 0, 0, 0, -1.85308724334741712e9,
-7.41234897339007950e9, -1.85308724334731412e9,
-0.0208333333333333738, 0, 0, 
 2.77963086502112484e9, -5.55926173004224968e9, 0, 0,
-0.0104166666666666054, 0.0208333333333332107, -0.0104166666666666054,
  7.49837536767640727e-14, -6.24999999999997060e-8, 0, 0, 0, 0,
-166.666666666666657, 0, 2.77963086502112484e9, 
 2.77963086502112484e9, 0, 0, -1.85308724334731412e9, 0,
-9.26543621673707724e8, -0.0104166666666666869, 0,
-1.85308724334731412e9, -9.26543621673707724e8, 0,
-0.0104166666666666869, 0, -0.0208333333333332107,
0.0104166666666666054, 0.0104166666666666054,
-8.33333333333329331e-17, -6.24999999999997060e-8, 0, 0, 0, 0,
  0};

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    if(fabs(gold_jac_entries[i]) < 1e-12){
      TEST_ASSERT(fabs(jac_entriesHost(i)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-13);
    }
  }
}

TEUCHOS_UNIT_TEST( PlatoMathFunctors, RowSumSolve )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;

  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", meshWidth);

  using ElementType = Plato::Stabilized::ThermomechanicsElement<Plato::Tet4>;
  constexpr auto spaceDim = ElementType::mNumSpatialDims;

  // create mesh based solution from host data
  //
  int tNumNodes = tMesh->NumNodes();
  Plato::ScalarVector tProjectState ("state",     tNumNodes);
  Plato::ScalarVector tProjPGrad    ("ProjPGrad", tNumNodes*spaceDim);
  Plato::ScalarVector tControl      ("Control",   tNumNodes);
  Plato::blas1::fill( 1.0, tControl );
  Plato::blas1::fill( 0.0, tProjPGrad );
  Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
  {
     tProjectState(aNodeOrdinal) = 1.0*aNodeOrdinal;
  });

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                      \n"
    "  <ParameterList name='Spatial Model'>                                                    \n"
    "    <ParameterList name='Domains'>                                                        \n"
    "      <ParameterList name='Design Volume'>                                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                      \n"
    "        <Parameter name='Material Model' type='string' value='Squeaky Cheese'/>           \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "  <ParameterList name='Material Models'>                                                  \n"
    "    <ParameterList name='Squeaky Cheese'>                                                 \n"
    "      <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>                        \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.499'/>                   \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>   \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "      </ParameterList>                                                                    \n"
    "    </ParameterList>                                                                      \n"
    "  </ParameterList>                                                                        \n"
    "</ParameterList>                                                                          \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *params, tDataMap);
  Plato::Stabilized::VectorFunction<::Plato::Stabilized::Projection<Plato::Tet4, ElementType::mNumDofsPerNode, ElementType::mPressureDofOffset>>
    tVectorFunction(tSpatialModel, tDataMap, *params, "State Gradient Projection");

  auto tResidual = tVectorFunction.value      (tProjPGrad, tProjectState, tControl);
  auto tJacobian = tVectorFunction.gradient_u (tProjPGrad, tProjectState, tControl);


  { // test residual
    //

    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar> tGold = {
      -0.5625000000000000, -0.1875000000000000,  -0.06249999999999998,
      -0.7500000000000000, -0.2500000000000000,  -0.08333333333333338,
      -0.1875000000000000, -0.06250000000000000, -0.02083333333333334,
      -0.7500000000000000, -0.2500000000000000,  -0.08333333333333329,
      -1.125000000000000,  -0.3750000000000000,  -0.1250000000000001,
      -0.3750000000000000, -0.1250000000000000,  -0.04166666666666669
    };
    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(i), tGold[i], 2.0e-14);
      }
    }
  }

  { // test jacobian
    //

    auto tJacobian_Host = Kokkos::create_mirror_view( tJacobian->entries() );
    Kokkos::deep_copy( tJacobian_Host, tJacobian->entries() );

    std::vector<Plato::Scalar> tGold = {
      0.0124999999999999053, 0, 0, 0, 0.0124999999999999053, 0, 0, 0,
      0.0124999999999999053, 0.00208333333333332333, 0, 0, 0,
      0.00208333333333332333, 0, 0, 0, 0.00208333333333332333,
      0.00208333333333332333, 0, 0, 0, 0.00208333333333332333, 0, 0, 0,
      0.00208333333333332333, 0.00208333333333332333, 0, 0, 0,
      0.00208333333333332333, 0, 0, 0, 0.00208333333333332333
    };

    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tJacobian_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tJacobian_Host(i), tGold[i], 2.0e-14);
      }
    }
  }


  { // test row sum functor
    //
    Plato::RowSum rowSum(tJacobian);

    Plato::ScalarVector tRowSum("row sum", tResidual.extent(0));

    auto tNumBlockRows = tJacobian->rowMap().size() - 1;
    Kokkos::parallel_for("row sum inverse", Kokkos::RangePolicy<int>(0,tNumBlockRows), KOKKOS_LAMBDA(int blockRowOrdinal)
    {
      // compute row sum
      rowSum(blockRowOrdinal, tRowSum);

    });

    auto tRowSum_Host = Kokkos::create_mirror_view( tRowSum );
    Kokkos::deep_copy( tRowSum_Host, tRowSum );

    std::vector<Plato::Scalar> tRowSum_gold = {
    0.0312500000000000000, 0.0312500000000000000, 0.0312500000000000000,
    0.0416666666666666713, 0.0416666666666666713, 0.0416666666666666713,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0625000000000000000, 0.0625000000000000000, 0.0625000000000000000,
    0.124999999999999958,  0.124999999999999958,  0.124999999999999958,
    0.0624999999999999931, 0.0624999999999999931, 0.0624999999999999931,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0624999999999999931, 0.0624999999999999931, 0.0624999999999999931,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0208333333333333322, 0.0208333333333333322, 0.0208333333333333322,
    0.0624999999999999931, 0.0624999999999999931, 0.0624999999999999931,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0104166666666666661, 0.0104166666666666661, 0.0104166666666666661,
    0.0416666666666666644, 0.0416666666666666644, 0.0416666666666666644,
    0.0312500000000000000, 0.0312500000000000000, 0.0312500000000000000
    };

    int tNumGold = tRowSum_gold.size();
    for(int i=0; i<tNumGold; i++){
      TEST_FLOATING_EQUALITY(tRowSum_Host(i), tRowSum_gold[i], 2.0e-14);
    }
  }


  { // test row summed solve
    //
    Plato::blas1::scale(-1.0, tResidual);
    Plato::Solve::RowSummed<spaceDim>(tJacobian, tProjPGrad, tResidual);

    auto tProjPGrad_Host = Kokkos::create_mirror_view( tProjPGrad );
    Kokkos::deep_copy( tProjPGrad_Host, tProjPGrad );

    std::vector<Plato::Scalar> tGold = {
      18.0000000000000000, 5.99999999999999911, 1.99999999999999933,
      17.9999999999999964, 5.99999999999999911, 2.00000000000000089,
      18.0000000000000000, 6.00000000000000000, 2.00000000000000000,
      18.0000000000000000, 6.00000000000000000, 1.99999999999999889,
      17.9999999999999964, 6.00000000000000000, 2.00000000000000133,
      18.0000000000000000, 6.00000000000000000, 2.00000000000000089,
      18.0000000000000000, 5.99999999999999822, 1.99999999999999956,
      17.9999999999999964, 6.00000000000000000, 2.00000000000000133
    };

    int tNumGold = tGold.size();
    for(int i=0; i<tNumGold; i++){
      if(tGold[i] == 0.0){
        TEST_ASSERT(fabs(tProjPGrad_Host(i)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tProjPGrad_Host(i), tGold[i], 2.0e-14);
      }
    }
  }
}
