#include "util/PlatoTestHelpers.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <sstream>
#include <fstream>
#include <stdio.h>

#include "PlatoStaticsTypes.hpp"

#include "Tet4.hpp"
#include "MechanicsElement.hpp"

#include "Plato_InputData.hpp"
#include "Plato_Exceptions.hpp"
#include "Plato_Parser.hpp"
#include "Plato_MeshMap.hpp"

#include "WorksetBase.hpp"
#include "ImplicitFunctors.hpp"
#include "SpatialModel.hpp"

#include "InterpolateFromNodal.hpp"

#include "WeightedNormalVector.hpp"
#include "SurfaceArea.hpp"

#include "elliptic/EvaluationTypes.hpp"

#include "ContactUtils.hpp"
#include "SurfaceDisplacement.hpp"
#include "ProjectedSurfaceDisplacement.hpp"

namespace ContactTests
{

template <typename EvaluationType>
class DummyResidual
{
private:
    using ElementType      = typename EvaluationType::ElementType;
    using StateScalarType  = typename EvaluationType::StateScalarType;  
    using ResultScalarType = typename EvaluationType::ResultScalarType; 

public:

    template <typename SurfaceDispType>
    void exercise_surface_disp_interface
    (const SurfaceDispType                             & aComputeSurfaceDisp,
     Plato::OrdinalType                                  aCellOrdinal, 
     const Plato::Array<ElementType::mNumNodesPerFace> & aBasisFunctions,
     const Plato::ScalarMultiVectorT<StateScalarType>  & aState,
           Plato::ScalarVectorT<ResultScalarType>      & aSurfaceDisp)
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal)
        {
            Plato::Array<ElementType::mNumSpatialDims, StateScalarType> tSurfaceDisp;
            aComputeSurfaceDisp(aCellOrdinal, aBasisFunctions, aState, tSurfaceDisp);
            for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
            {
                Kokkos::atomic_add(&aSurfaceDisp(tDof), tSurfaceDisp(tDof));
            }
        }, "do it on device");
        }

    template <typename SurfaceDispType>
    void dummy_contact_force
    (const Plato::SpatialModel                         & aSpatialModel,
     const std::string                                 & aSideSet,
     const Plato::ScalarMultiVectorT<StateScalarType>  & aState,
     const SurfaceDispType                             & aComputeSurfaceDisp,
           Plato::ScalarMultiVectorT<ResultScalarType> & aResult)
    {
        auto tElementOrds   = aSpatialModel.Mesh->GetSideSetElements(aSideSet);
        auto tLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(aSideSet);
        Plato::OrdinalType tNumFaces = tElementOrds.size();

        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tCubatureWeights = ElementType::Face::getCubWeights();
        auto tNumPoints = tCubatureWeights.size();

        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tCubaturePoint = tCubaturePoints(iGPOrdinal);
            auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);

            Plato::Array<ElementType::mNumSpatialDims, StateScalarType> tSurfaceDisp;
            aComputeSurfaceDisp(iCellOrdinal, tBasisValues, aState, tSurfaceDisp);

            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
            {
                auto tLocalNodeOrd = tLocalNodeOrds(iCellOrdinal*ElementType::mNumNodesPerFace+tNode);

                for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
                {
                    auto tElementDofOrdinal = tLocalNodeOrd * ElementType::mNumDofsPerNode + tDof;
                    ResultScalarType tResult = tBasisValues(tNode)*tSurfaceDisp[tDof];
                    Kokkos::atomic_add(&aResult(iCellOrdinal, tElementDofOrdinal), tResult);
                }
            }

        }, "contact force");
    }

};

Plato::SpatialModel
setup_dummy_spatial_model(Plato::Mesh aMesh)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                           \n"
        "  <ParameterList name='Spatial Model'>                                         \n"
        "    <ParameterList name='Domains'>                                             \n"
        "      <ParameterList name='Design Volume'>                                     \n"
        "        <Parameter name='Element Block' type='string' value='body'/>           \n"
        "        <Parameter name='Material Model' type='string' value='Fancy Feast'/>   \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "  <ParameterList name='Material Models'>                                       \n"
        "    <ParameterList name='Fancy Feast'>                                         \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                          \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.35'/>         \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>       \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "</ParameterList>                                                               \n"
      );

    Plato::DataMap tDataMap;
    return Plato::SpatialModel(aMesh, *tInputs, tDataMap);
}

Teuchos::RCP<Teuchos::ParameterList>
get_2box_mesh_params()
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                           \n"
        "  <ParameterList name='Spatial Model'>                                         \n"
        "    <ParameterList name='Domains'>                                             \n"
        "      <ParameterList name='Box 1'>                                             \n"
        "        <Parameter name='Element Block' type='string' value='block_1'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Fancy Feast'/>   \n"
        "      </ParameterList>                                                         \n"
        "      <ParameterList name='Box 2'>                                             \n"
        "        <Parameter name='Element Block' type='string' value='block_2'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Fancy Feast'/>   \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"

        "  <ParameterList name='Contact'>                                                     \n"
        "    <ParameterList name='Pairs'>                                                     \n"
        "      <ParameterList name='Pair 1'>                                                  \n"
        "        <Parameter name='Side A Block'  type='string' value='block_1'/>       \n"
        "        <Parameter name='Side A Child Sideset' type='string' value='block1_child'/>  \n"
        "        <Parameter name='Side B Block'  type='string' value='block_2'/>       \n"
        "        <Parameter name='Side B Child Sideset' type='string' value='block2_child'/>  \n"
        "        <Parameter name='Initial Gap' type='Array(double)' value='{1.0,0.0,0.0}' />  \n"
        "      </ParameterList>                                                               \n"
        "    </ParameterList>                                                                 \n"
        "  </ParameterList>                                                                   \n"

        "  <ParameterList name='Material Models'>                                       \n"
        "    <ParameterList name='Fancy Feast'>                                         \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                          \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.35'/>         \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>       \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "</ParameterList>                                                               \n"
      );

    return tInputs;
}

TEUCHOS_UNIT_TEST(ParsingTests, ParseFromTeuchosParameterList)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    auto tContactParams = tInputs->sublist("Contact");
    auto tPairsParams = tContactParams.sublist("Pairs");

    // test parsing number of pairs
    Plato::OrdinalType tNumPairs(0);
    for (auto tIndex = tPairsParams.begin(); tIndex != tPairsParams.end(); ++tIndex)
        tNumPairs++;
    TEST_EQUALITY(tNumPairs, 1);

    // parse pair data
    const auto& tMyName = tPairsParams.name(tPairsParams.begin());
    Teuchos::ParameterList& tPairParams = tPairsParams.sublist(tMyName);
    TEST_EQUALITY(tPairParams.get<std::string>("Side A Block"), "block_1");
    TEST_EQUALITY(tPairParams.get<std::string>("Side A Child Sideset"), "block1_child");
    TEST_EQUALITY(tPairParams.get<std::string>("Side B Block"), "block_2");
    TEST_EQUALITY(tPairParams.get<std::string>("Side B Child Sideset"), "block2_child");
}

TEUCHOS_UNIT_TEST(UtilsTests, ParseSingleContactPair)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    auto tContactParams = tInputs->sublist("Contact");
    auto tPairsParams = tContactParams.sublist("Pairs");
    const auto& tMyName = tPairsParams.name(tPairsParams.begin());
    Teuchos::ParameterList& tPairParams = tPairsParams.sublist(tMyName);

    Plato::Contact::ContactPair tPair = Plato::Contact::parse_contact_pair(tPairParams, tMesh);

    // test child nodes
    auto tSideAChild = tPair.childNodesA;
    TEST_EQUALITY(tSideAChild.size(), 4);

    auto tSideAChild_Host = Plato::TestHelpers::get( tSideAChild );
    std::vector<Plato::OrdinalType> tSideAChild_Gold = {0, 5, 6, 7};
    for(int iVal=0; iVal<tSideAChild_Gold.size(); iVal++){
        TEST_EQUALITY(tSideAChild_Host(iVal), tSideAChild_Gold[iVal]);
    }

    auto tSideBChild = tPair.childNodesB;
    TEST_EQUALITY(tSideBChild.size(), 4);

    auto tSideBChild_Host = Plato::TestHelpers::get( tSideBChild );
    std::vector<Plato::OrdinalType> tSideBChild_Gold = {9, 10, 11, 12};
    for(int iVal=0; iVal<tSideBChild_Gold.size(); iVal++){
        TEST_EQUALITY(tSideBChild_Host(iVal), tSideBChild_Gold[iVal]);
    }

    // test initial gap
    std::vector<Plato::Scalar> tInitialGap_Gold = {1.0, 0.0, 0.0};
    for(int iVal=0; iVal<tInitialGap_Gold.size(); iVal++){
        TEST_EQUALITY(tPair.initialGap[iVal], tInitialGap_Gold[iVal]);
    }
}

TEUCHOS_UNIT_TEST(UtilsTests, ParseAllContactPairs)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);

    // test number of pairs
    TEST_EQUALITY(tPairs.size(), 1);
}

TEUCHOS_UNIT_TEST(UtilsTests, PopulateFullContactArrays)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tElementType = tMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);

    auto tNumTotalNodes = Plato::Contact::count_total_child_nodes(tPairs);

    // test total number of child nodes
    TEST_EQUALITY(tNumTotalNodes, 8);

    Plato::OrdinalVector tAllChildNodes("", tNumTotalNodes);
    Plato::OrdinalVector tAllParentElements("", tNumTotalNodes);
    Plato::Contact::populate_full_contact_arrays<ElementType>(tPairs, tSpatialModel, tAllChildNodes, tAllParentElements);
    Plato::Contact::check_for_repeated_child_nodes(tAllChildNodes,tMesh);

    // test child nodes
    auto tAllChildNodes_Host = Plato::TestHelpers::get( tAllChildNodes );
    std::vector<Plato::OrdinalType> tAllChildNodes_Gold = {0, 5, 6, 7, 9, 10, 11, 12};
    for(int iVal=0; iVal<tAllChildNodes_Gold.size(); iVal++){
        TEST_EQUALITY(tAllChildNodes_Host(iVal), tAllChildNodes_Gold[iVal]);
    }

    // test parent elements
    auto tAllParentElements_Host = Plato::TestHelpers::get( tAllParentElements );
    std::vector<Plato::OrdinalType> tAllParentElements_Gold = {7, 6, 6, 6, 4, 2, 2, 0};
    for(int iVal=0; iVal<tAllParentElements_Gold.size(); iVal++){
        TEST_EQUALITY(tAllParentElements_Host(iVal), tAllParentElements_Gold[iVal]);
    }
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_ChildElementContrbution)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tElementType = tMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")

    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);
     
    // get contact pair A face info
    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);
    auto tPair = tPairs[0]; // there is only 1 pair

    // get basis functions at cubature (quadrature, since surface) point
    Plato::OrdinalType tCubOrdinal = 0;
    auto tCubPoints = ElementType::Face::getCubPoints();
    auto tCubPoint = tCubPoints(tCubOrdinal);
    auto tBasisValues = ElementType::Face::basisValues(tCubPoint);

    // test child elements
    auto tChildElements_Host = Plato::TestHelpers::get( tPair.childElementsA );
    std::vector<Plato::OrdinalType> tChildElements_gold = { 2, 4 };
    for(int iChild=0; iChild<int(tChildElements_gold.size()); iChild++){
        TEST_EQUALITY(tChildElements_Host(iChild), tChildElements_gold[iChild]);
    }

    // construct dummy residual class
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    DummyResidual<EvaluationType> tResidual;

    // construct compute surface displacement functor
    Plato::Contact::SurfaceDisplacement<EvaluationType> tComputeSurfaceDisp(tPair.childElementsA, tPair.childFaceLocalNodesA, -1.0);

    // test surface displacement child face cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;
    Plato::ScalarVector tSurfaceDisp0("make on device", ElementType::mNumDofsPerNode);
    tResidual.exercise_surface_disp_interface(tComputeSurfaceDisp, tChildCellOrdinal, tBasisValues, tDispWS, tSurfaceDisp0);

    auto tSurfaceDisp0_Host = Plato::TestHelpers::get( tSurfaceDisp0 );
    std::vector<double> tSurfaceDisp_Gold = {-0.0012, -0.0013, -0.0014};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp0_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }

    // test surface displacement child face cell 1
    tChildCellOrdinal = 1;
    Plato::ScalarVector tSurfaceDisp1("make on device", ElementType::mNumDofsPerNode);
    tResidual.exercise_surface_disp_interface(tComputeSurfaceDisp, tChildCellOrdinal, tBasisValues, tDispWS, tSurfaceDisp1);

    auto tSurfaceDisp1_Host = Plato::TestHelpers::get( tSurfaceDisp1 );
    tSurfaceDisp_Gold = {-0.0013, -0.0014, -0.0015};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp1_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_SingleParentElementContribution)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tElementType = tMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")
    
    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // get contact pair A face info
    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);
    auto tPair = tPairs[0]; // there is only 1 pair

    auto tChildFaceElements = tPair.childElementsA;
    auto tNumChildCells = tChildFaceElements.extent(0);
    auto tChildFaceLocalNodes = tPair.childFaceLocalNodesA;
     
    // construct dummy residual class
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    DummyResidual<EvaluationType> tResidual;

    // construct compute surface displacement functor
    auto tGlobalLocalChildNodeOrdMap = Plato::Contact::global_local_child_node_ord_map(tPair.childNodesA, tMesh->NumNodes());
    auto tElementWiseChildNodeOrdMap = Plato::Contact::convert_to_elementwise_map(tPair.childElementsA, tPair.childFaceLocalNodesA, tGlobalLocalChildNodeOrdMap, tMesh, ElementType::mNumNodesPerFace);
    
    auto tChildLocations = Plato::Contact::compute_node_locations(tSpatialModel.Mesh, tPair.childNodesA);
    auto tMappedChildLocations = Plato::Contact::map_node_locations(tChildLocations, tPair.initialGap);
    Plato::SpatialDomain tDomain = Plato::Contact::get_domain(tPair.parentBlockB, tSpatialModel.Domains);

    Plato::OrdinalVector tParentElements("parent elements", tPair.childNodesA.size());
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
    (tSpatialModel.Mesh, tDomain.cellOrdinals(), tChildLocations, tMappedChildLocations, tParentElements);

    Plato::Contact::ProjectedSurfaceDisplacement<EvaluationType> tComputeSurfaceDisp(tParentElements, tMappedChildLocations, tElementWiseChildNodeOrdMap, tMesh);

    // test surface displacement terms for each child node on child cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;

    // get basis functions at cubature (quadrature, since surface) point
    Plato::OrdinalType tCubOrdinal = 0;
    auto tCubPoints = ElementType::Face::getCubPoints();
    auto tCubPoint = tCubPoints(tCubOrdinal);
    auto tBasisValues = ElementType::Face::basisValues(tCubPoint);

    std::vector<std::vector<double>> tSurfaceDisp_Gold = {
        {0.0037 / 3.0, 0.0038 / 3.0, 0.0039 / 3.0},
        {0.0034 / 3.0, 0.0035 / 3.0, 0.0036 / 3.0},
        {0.0031 / 3.0, 0.0032 / 3.0, 0.0033 / 3.0}
    };

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        Plato::ScalarVector tSurfaceDisp("make on device", ElementType::mNumDofsPerNode);
        
        tComputeSurfaceDisp.setChildNode(iChildNode);
        tResidual.exercise_surface_disp_interface(tComputeSurfaceDisp, tChildCellOrdinal, tBasisValues, tDispWS, tSurfaceDisp);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( tSurfaceDisp );
        for(int iDof=0; iDof<tSurfaceDisp_Gold[iChildNode].size(); iDof++){
            TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iChildNode][iDof], 1e-12);
        }
    }

    // test surface displacement terms for each child node on child cell 1
    tChildCellOrdinal = 1;

    tSurfaceDisp_Gold = {
        {0.0037 / 3.0, 0.0038 / 3.0, 0.0039 / 3.0},
        {0.0031 / 3.0, 0.0032 / 3.0, 0.0033 / 3.0},
        {0.0028 / 3.0, 0.0029 / 3.0, 0.0030 / 3.0}
    };

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        Plato::ScalarVector tSurfaceDisp("make on device", ElementType::mNumDofsPerNode);

        tComputeSurfaceDisp.setChildNode(iChildNode);
        tResidual.exercise_surface_disp_interface(tComputeSurfaceDisp, tChildCellOrdinal, tBasisValues, tDispWS, tSurfaceDisp);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( tSurfaceDisp );
        for(int iDof=0; iDof<tSurfaceDisp_Gold[iChildNode].size(); iDof++){
            TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iChildNode][iDof], 1e-12);
        }
    }
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_LoopThroughContributions)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tElementType = tMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")
    
    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // construct dummy residual class
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    DummyResidual<EvaluationType> tResidual;

    // construct compute surface displacement functors
    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);
    auto tPair = tPairs[0]; // there is only 1 pair
    Plato::Contact::SurfaceDisplacement<EvaluationType> computeChildSurfaceDisp(tPair.childElementsA, tPair.childFaceLocalNodesA, -1.0);

    auto tGlobalLocalChildNodeOrdMap = Plato::Contact::global_local_child_node_ord_map(tPair.childNodesA, tMesh->NumNodes());
    auto tElementWiseChildNodeOrdMap = Plato::Contact::convert_to_elementwise_map(tPair.childElementsA, tPair.childFaceLocalNodesA, tGlobalLocalChildNodeOrdMap, tMesh, ElementType::mNumNodesPerFace);
    
    auto tChildLocations = Plato::Contact::compute_node_locations(tSpatialModel.Mesh, tPair.childNodesA);
    auto tMappedChildLocations = Plato::Contact::map_node_locations(tChildLocations, tPair.initialGap);
    Plato::SpatialDomain tDomain = Plato::Contact::get_domain(tPair.parentBlockB, tSpatialModel.Domains);

    Plato::OrdinalVector tParentElements("parent elements", tPair.childNodesA.size());
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
    (tSpatialModel.Mesh, tDomain.cellOrdinals(), tChildLocations, tMappedChildLocations, tParentElements);

    Plato::Contact::ProjectedSurfaceDisplacement<EvaluationType> computeParentSurfaceDisp(tParentElements, tMappedChildLocations, tElementWiseChildNodeOrdMap, tMesh);

    // test computation of displacement difference (dummy contact force)
    Plato::ScalarMultiVectorT<Plato::Scalar> tResult("dummy contact force", tPair.childElementsA.size(), ElementType::mNumDofsPerCell);
    tResidual.dummy_contact_force(tSpatialModel,tPair.childSideSetA,tDispWS,computeChildSurfaceDisp,tResult); // child face contributions

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        computeParentSurfaceDisp.setChildNode(iChildNode);
        tResidual.dummy_contact_force(tSpatialModel,tPair.childSideSetA,tDispWS,computeParentSurfaceDisp,tResult); // parent face contributions
    }

    std::vector<std::vector<double>> tResult_Gold = {
        {0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0, 0.0, 0.0},
        {0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0, 0.0, 0.0}
    };

    auto tResult_Host = Plato::TestHelpers::get( tResult );

    for(int iCell=0; iCell<int(tPair.childElementsA.size()); iCell++){
        for(int iDof=0; iDof<ElementType::mNumNodesPerFace*ElementType::mNumDofsPerNode; iDof++){
            TEST_FLOATING_EQUALITY(tResult_Host(iCell,iDof), tResult_Gold[iCell][iDof], 1e-12);
      }
    }

}

}

