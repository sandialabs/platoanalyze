#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Tet4.hpp"
#include "WorksetBase.hpp"
#include "SurfaceStateIntegral.hpp"
#include "ThermomechanicsElement.hpp"

TEUCHOS_UNIT_TEST( NaturalBCTests, SurfaceStateIntegral )
{ 
  Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                           \n"
    "  <ParameterList name='Spatial Model'>                                         \n"
    "    <ParameterList name='Domains'>                                             \n"
    "      <ParameterList name='Design Volume'>                                     \n"
    "        <Parameter name='Element Block' type='string' value='body'/>           \n"
    "        <Parameter name='Material Model' type='string' value='Material'/>      \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "  <ParameterList name='Material Models'>                                       \n"
    "    <ParameterList name='Material'>                                            \n"
    "      <ParameterList name='Thermoelastic'>                                     \n"
    "        <ParameterList name='Elastic Stiffness'>                               \n"
    "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>     \n"
    "        </ParameterList>                                                       \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>  \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='910.0'/>  \n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>   \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );

  // create test mesh
  //
  constexpr int tMeshWidth=4;

  auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

  using ElementType = typename Plato::ThermomechanicsElement<Plato::Tet4>;

  int tNumCells = tMesh->NumElements();
  int tNumNodes = tMesh->NumNodes();
  
  constexpr int cNumDofs = 1;
  constexpr int cNumSpaceDims = ElementType::mNumSpatialDims;
  constexpr int cDofsPerNode  = ElementType::mNumDofsPerNode;
  constexpr int cDofsPerCell  = ElementType::mNumDofsPerCell;
  constexpr int cNodesPerCell = ElementType::mNumNodesPerCell;
  constexpr int cNodesPerFace = ElementType::Face::mNumNodesPerCell;
  constexpr int cTemperatureDofIndex = cNumSpaceDims;
  constexpr int cDofOffset    = cNumSpaceDims;

  std::vector<std::string> tStateNames = {"T"};
  std::vector<std::string> tFluxExpressions = {"pow(T,4)"};

  std::string tOnFace = "x+";
  Plato::SurfaceStateIntegral<ElementType, cNumDofs, cDofsPerNode, cDofOffset>
    tSurfaceStateIntegral(tOnFace, tFluxExpressions, tStateNames);

  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(tMesh, *tParams, tDataMap);

  Plato::ScalarMultiVector tState("state", tNumCells, cDofsPerCell);
  Kokkos::deep_copy(tState, 2.0);

  Plato::ScalarMultiVector tControl("control", tNumCells, cNodesPerCell);
  Kokkos::deep_copy(tControl, 1.0);

  Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
  Plato::ScalarArray3D tConfig("config workset", tNumCells, cNodesPerCell, cNumSpaceDims);
  tWorksetBase.worksetConfig(tConfig);

  Plato::ScalarMultiVector tResult("control", tNumCells, cDofsPerCell);

  Plato::Scalar tScale(1.0);
  tSurfaceStateIntegral(tSpatialModel, tState, tControl, tConfig, tResult, tScale);

  auto tResult_Host = Kokkos::create_mirror_view(tResult);
  Kokkos::deep_copy(tResult_Host, tResult);

  auto tConfig_Host = Kokkos::create_mirror_view(tConfig);
  Kokkos::deep_copy(tConfig_Host, tConfig);

  auto tElementOrds = tMesh->GetSideSetElements(tOnFace);
  auto tElementOrds_Host = Kokkos::create_mirror_view(tElementOrds);
  Kokkos::deep_copy(tElementOrds_Host, tElementOrds);

  auto tNodeOrds = tMesh->GetSideSetLocalNodes(tOnFace);
  auto tNodeOrds_Host = Kokkos::create_mirror_view(tNodeOrds);
  Kokkos::deep_copy(tNodeOrds_Host, tNodeOrds);

  int tNumFacePoints = 2.0*tMeshWidth*tMeshWidth*cNodesPerFace;
  Plato::Scalar tNodeVal = pow(2.0,4)/tNumFacePoints;

  int tNumFacePointsFound = 0;
  for(int iElem=0; iElem<tElementOrds_Host.size(); iElem++)
  {
    auto tElemIndex = tElementOrds_Host(iElem);
    for(int iNode=0; iNode<cNodesPerFace; iNode++)
    {
      auto tElemLocalNodeIndex = tNodeOrds_Host(iElem*cNodesPerFace+iNode);
      tNumFacePointsFound++;
      TEST_FLOATING_EQUALITY(tConfig_Host(tElemIndex, tElemLocalNodeIndex, 0), 1.0,  DBL_EPSILON);
      TEST_FLOATING_EQUALITY(tResult_Host(tElemIndex, tElemLocalNodeIndex*cDofsPerNode+cTemperatureDofIndex), tNodeVal, 2.0*DBL_EPSILON);
    }
  }
  TEST_ASSERT( tNumFacePointsFound == tNumFacePoints );
  
}
