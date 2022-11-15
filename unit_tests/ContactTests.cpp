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
#include "SpatialModel.hpp"

#include "InterpolateFromNodal.hpp"

#include "WeightedNormalVector.hpp"
#include "SurfaceArea.hpp"

#include "elliptic/EvaluationTypes.hpp"

#include "ContactPair.hpp"
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
    void dummy_contact_force
    (const Plato::SpatialModel                                         & aSpatialModel,
     const std::string                                                 & aSideSet,
     const Plato::ScalarMultiVectorT<StateScalarType>                  & aState,
     const Plato::Contact::AbstractSurfaceDisplacement<EvaluationType> & aComputeSurfaceDisp,
           Plato::ScalarMultiVectorT<ResultScalarType>                 & aResult)
    {
        auto tElementOrds   = aSpatialModel.Mesh->GetSideSetElements(aSideSet);

        Plato::ScalarMultiVectorT<ResultScalarType> tSurfaceDisplacement("", tElementOrds.size(), ElementType::mNumDofsPerNode);
        aComputeSurfaceDisp(tElementOrds, aState, tSurfaceDisplacement);

        Plato::OrdinalType tNumFaces = tElementOrds.size();

        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tCubatureWeights = ElementType::Face::getCubWeights();
        auto tNumPoints = tCubatureWeights.size();

        auto tLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(aSideSet);

        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tCubaturePoint = tCubaturePoints(iGPOrdinal);
            auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);

            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
            {
                auto tLocalNodeOrd = tLocalNodeOrds(iCellOrdinal*ElementType::mNumNodesPerFace+tNode);

                for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
                {
                    auto tElementDofOrdinal = tLocalNodeOrd * ElementType::mNumDofsPerNode + tDof;
                    ResultScalarType tResult = tBasisValues(tNode)*tSurfaceDisplacement(iCellOrdinal, tDof);
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

void
check_element_type_is_tet(Plato::Mesh aMesh)
{
    auto tElementType = aMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")
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
        "        <Parameter name='Initial Gap' type='Array(double)' value='{1.0,0.0,0.0}' />  \n"
        // "        <Parameter name='Penalty Value' type='Array(double)' value='{1.0e4,1.0e4,1.0e4}' />  \n"
        // "        <Parameter name='Penalty Type' type='string' value='tensor' />  \n"
        "        <Parameter name='Penalty Value' type='double' value='1.0e4' />  \n"
        "        <Parameter name='Penalty Type' type='string' value='normal' />  \n"
        "        <ParameterList name='A Surface'>                                                  \n"
        "          <Parameter name='Child Sideset' type='string' value='block1_child'/>  \n"
        "          <Parameter name='Parent Block'  type='string' value='block_2'/>       \n"
        "        </ParameterList>                                                               \n"
        "        <ParameterList name='B Surface'>                                                  \n"
        "          <Parameter name='Child Sideset' type='string' value='block2_child'/>  \n"
        "          <Parameter name='Parent Block'  type='string' value='block_1'/>       \n"
        "        </ParameterList>                                                               \n"
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

TEUCHOS_UNIT_TEST(UtilsTests, ParseSingleContactPair)
{
    Teuchos::RCP<Teuchos::ParameterList> tContactParams =
        Teuchos::getParametersFromXmlString(
        "  <ParameterList name='Contact'>                                                     \n"
        "    <ParameterList name='Pairs'>                                                     \n"
        "      <ParameterList name='Pair 1'>                                                  \n"
        "        <Parameter name='Initial Gap' type='Array(double)' value='{1.0,0.0,0.0}' />  \n"
        "        <Parameter name='Penalty Value' type='double' value='1.0e4' />  \n"
        "        <Parameter name='Penalty Type' type='string' value='normal' />  \n"
        "        <ParameterList name='A Surface'>                                                  \n"
        "          <Parameter name='Child Sideset' type='string' value='block1_child'/>  \n"
        "          <Parameter name='Parent Block'  type='string' value='block_2'/>       \n"
        "        </ParameterList>                                                               \n"
        "        <ParameterList name='B Surface'>                                                  \n"
        "          <Parameter name='Child Sideset' type='string' value='block2_child'/>  \n"
        "          <Parameter name='Parent Block'  type='string' value='block_1'/>       \n"
        "        </ParameterList>                                                               \n"
        "      </ParameterList>                                                               \n"
        "    </ParameterList>                                                                 \n"
        "  </ParameterList>                                                                   \n"
      );

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    auto tPairsParams = tContactParams->sublist("Pairs");
    const auto& tMyName = tPairsParams.name(tPairsParams.begin());
    Teuchos::ParameterList& tPairParams = tPairsParams.sublist(tMyName);

    Plato::Contact::ContactPair tPair = Plato::Contact::parse_contact_pair(tPairParams, tMesh);

    // test child nodes
    auto tSideAChild = tPair.surfaceA.childNodes();
    TEST_EQUALITY(tSideAChild.size(), 4);

    auto tSideAChild_Host = Plato::TestHelpers::get( tSideAChild );
    std::vector<Plato::OrdinalType> tSideAChild_Gold = {0, 5, 6, 7};
    for(int iVal=0; iVal<tSideAChild_Gold.size(); iVal++){
        TEST_EQUALITY(tSideAChild_Host(iVal), tSideAChild_Gold[iVal]);
    }

    auto tSideBChild = tPair.surfaceB.childNodes();
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

    // test penalty data
    TEST_EQUALITY(tPair.penaltyType, "normal");
    TEST_EQUALITY(tPair.penaltyValue.size(), 1);
    TEST_FLOATING_EQUALITY(tPair.penaltyValue[0], 1.0e4, 1e-13)
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

    check_element_type_is_tet(tMesh);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);

    auto tNumTotalNodes = Plato::Contact::count_total_child_nodes(tPairs);

    // test total number of child nodes
    TEST_EQUALITY(tNumTotalNodes, 8);

    Plato::OrdinalVector tAllChildNodes("", tNumTotalNodes);
    Plato::OrdinalVector tAllParentElements("", tNumTotalNodes);
    Plato::Contact::populate_full_contact_arrays(tPairs, tAllChildNodes, tAllParentElements);
    Plato::Contact::check_for_repeated_child_nodes(tAllChildNodes,tMesh->NumNodes());

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

TEUCHOS_UNIT_TEST(UtilsTests, CheckForRepeatedChildNodes)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // create artificial all child nodes vector with repeated entry
    std::vector<Plato::OrdinalType> tAllChildNodes = {1, 5, 6, 1};
    auto dAllChildNodes = Plato::TestHelpers::create_device_view(tAllChildNodes);

    // test
    TEST_THROW(Plato::Contact::check_for_repeated_child_nodes(dAllChildNodes, tMesh->NumNodes()), std::runtime_error);
}

TEUCHOS_UNIT_TEST(ContactSurfaceTests, InitialAssignmentOfParentDataIsPersistent)
{
    // add initial parent data
    std::vector<Plato::OrdinalType> tParentElements = {1, 5, 6, 3};
    auto dParentElements = Plato::TestHelpers::create_device_view(tParentElements);

    Plato::OrdinalVector tElementWiseChildMap;
    Plato::ScalarMultiVector tMappedChildNodeLocations;

    Plato::Contact::ContactSurface tSurface;
    tSurface.addParentData(dParentElements, tElementWiseChildMap, tMappedChildNodeLocations);

    // change parent elements and add again
    std::vector<Plato::OrdinalType> tNewParentElements = {8, 4, 1, 9};
    auto dNewParentElements = Plato::TestHelpers::create_device_view(tNewParentElements);
    tSurface.addParentData(dNewParentElements, tElementWiseChildMap, tMappedChildNodeLocations);

    // test that original parent elements weren't changed
    auto tStoredParentElements = tSurface.parentElements();
    auto tStoredParentElements_Host = Plato::TestHelpers::get( tStoredParentElements );
    for(int iOrd=0; iOrd<int(tParentElements.size()); iOrd++){
        TEST_EQUALITY(tStoredParentElements_Host(iOrd), tParentElements[iOrd]);
    }
}

TEUCHOS_UNIT_TEST(ContactSurfaceTests, ThrowWhenAccessingParentDataIfNotSet)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);
    auto tPair = tPairs[0];

    TEST_THROW(tPair.surfaceA.parentElements(), std::runtime_error);
    TEST_THROW(tPair.surfaceA.elementWiseChildMap(), std::runtime_error);
    TEST_THROW(tPair.surfaceA.mappedChildNodeLocations(), std::runtime_error);

    TEST_THROW(tPair.surfaceB.parentElements(), std::runtime_error);
    TEST_THROW(tPair.surfaceB.elementWiseChildMap(), std::runtime_error);
    TEST_THROW(tPair.surfaceB.mappedChildNodeLocations(), std::runtime_error);
}

TEUCHOS_UNIT_TEST(FunctorTests, ApplyContactPenalty_DiagonalMatrix)
{
    Teuchos::RCP<Teuchos::ParameterList> tContactParams =
        Teuchos::getParametersFromXmlString(
        "  <ParameterList name='Contact'>                                                     \n"
        "    <ParameterList name='Pairs'>                                                     \n"
        "      <ParameterList name='Pair 1'>                                                  \n"
        "        <Parameter name='Initial Gap' type='Array(double)' value='{1.0,0.0,0.0}' />  \n"
        "        <Parameter name='Penalty Type' type='string' value='tensor' />  \n"
        "        <Parameter name='Penalty Value' type='Array(double)' value='{1.0e5,1.0e5,1.0e5}' />  \n"
        "        <ParameterList name='A Surface'>                                                  \n"
        "          <Parameter name='Child Sideset' type='string' value='block1_child'/>  \n"
        "          <Parameter name='Parent Block'  type='string' value='block_2'/>       \n"
        "        </ParameterList>                                                               \n"
        "        <ParameterList name='B Surface'>                                                  \n"
        "          <Parameter name='Child Sideset' type='string' value='block2_child'/>  \n"
        "          <Parameter name='Parent Block'  type='string' value='block_1'/>       \n"
        "        </ParameterList>                                                               \n"
        "      </ParameterList>                                                               \n"
        "    </ParameterList>                                                                 \n"
        "  </ParameterList>                                                                   \n"
      );

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    check_element_type_is_tet(tMesh);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    auto tPairsParams = tContactParams->sublist("Pairs");
    const auto& tMyName = tPairsParams.name(tPairsParams.begin());
    Teuchos::ParameterList& tPairParams = tPairsParams.sublist(tMyName);

    Plato::Contact::ContactPair tPair = Plato::Contact::parse_contact_pair(tPairParams, tMesh);

    // apply contact penalty
    Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tProjectedDisp{45.3, 66.54, 77.88};
    Plato::Contact::ApplyContactPenalty<ElementType> applyContactPenalty(tPair.penaltyValue);
    Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tPenalizedDisp;
    applyContactPenalty(tProjectedDisp, tPenalizedDisp);

    // test
    std::vector<Plato::Scalar> tPenalizedDisp_Gold = {45.3e5, 66.54e5, 77.88e5};
    TEST_FLOATING_EQUALITY(tPenalizedDisp(0), tPenalizedDisp_Gold[0], 1.0e-13);
    TEST_FLOATING_EQUALITY(tPenalizedDisp(1), tPenalizedDisp_Gold[1], 1.0e-13);
    TEST_FLOATING_EQUALITY(tPenalizedDisp(2), tPenalizedDisp_Gold[2], 1.0e-13);
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_ChildElementContrbution)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    check_element_type_is_tet(tMesh);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

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
     
    // get contact pair info
    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);
    auto tPair = tPairs[0]; // there is only 1 pair

    // test child elements
    auto tChildElements_Host = Plato::TestHelpers::get( tPair.surfaceA.childElements() );
    std::vector<Plato::OrdinalType> tChildElements_gold = { 2, 4 };
    for(int iChild=0; iChild<int(tChildElements_gold.size()); iChild++){
        TEST_EQUALITY(tChildElements_Host(iChild), tChildElements_gold[iChild]);
    }

    tChildElements_Host = Plato::TestHelpers::get( tPair.surfaceB.childElements() );
    tChildElements_gold = { 6, 7 };
    for(int iChild=0; iChild<int(tChildElements_gold.size()); iChild++){
        TEST_EQUALITY(tChildElements_Host(iChild), tChildElements_gold[iChild]);
    }

    // construct compute surface displacement functors
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::Contact::SurfaceDisplacement<EvaluationType> tComputeSurfaceDispA(tPair.surfaceA.childFaceLocalNodes(), -1.0);
    Plato::Contact::SurfaceDisplacement<EvaluationType> tComputeSurfaceDispB(tPair.surfaceB.childFaceLocalNodes(), -1.0);

    // compute surface displacement for all child face cells
    Plato::ScalarMultiVector tSurfaceDispA("make on device", tPair.surfaceA.childElements().size(), ElementType::mNumDofsPerNode);
    tComputeSurfaceDispA(tPair.surfaceA.childElements(), tDispWS, tSurfaceDispA);

    Plato::ScalarMultiVector tSurfaceDispB("make on device", tPair.surfaceB.childElements().size(), ElementType::mNumDofsPerNode);
    tComputeSurfaceDispB(tPair.surfaceB.childElements(), tDispWS, tSurfaceDispB);

    // test surface displacement child face cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;

    auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispA, tChildCellOrdinal, Kokkos::ALL()) );
    std::vector<double> tSurfaceDisp_Gold = {-0.0012, -0.0013, -0.0014};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }

    tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispB, tChildCellOrdinal, Kokkos::ALL()) );
    tSurfaceDisp_Gold = {-0.0031, -0.0032, -0.0033};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }

    // test surface displacement child face cell 1
    tChildCellOrdinal = 1;

    tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispA, tChildCellOrdinal, Kokkos::ALL()) );
    tSurfaceDisp_Gold = {-0.0013, -0.0014, -0.0015};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }

    tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispB, tChildCellOrdinal, Kokkos::ALL()) );
    tSurfaceDisp_Gold = {-0.0033, -0.0034, -0.0035};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_SingleParentElementContribution)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    check_element_type_is_tet(tMesh);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
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

    // get contact pair info
    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);
    auto tPair = tPairs[0]; // there is only 1 pair

    // construct compute surface displacement functors
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::Contact::ProjectedSurfaceDisplacement<EvaluationType> tComputeSurfaceDispA(tPair.surfaceA.parentElements(), tPair.surfaceA.mappedChildNodeLocations(), tPair.surfaceA.elementWiseChildMap(), tMesh);
    Plato::Contact::ProjectedSurfaceDisplacement<EvaluationType> tComputeSurfaceDispB(tPair.surfaceB.parentElements(), tPair.surfaceB.mappedChildNodeLocations(), tPair.surfaceB.elementWiseChildMap(), tMesh);

    // test surface displacement terms for each child node on child cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;

    std::vector<std::vector<double>> tSurfaceDisp_Gold = {
        {0.0037 / 3.0, 0.0038 / 3.0, 0.0039 / 3.0},
        {0.0034 / 3.0, 0.0035 / 3.0, 0.0036 / 3.0},
        {0.0031 / 3.0, 0.0032 / 3.0, 0.0033 / 3.0}
    };

    Plato::ScalarVector tSurfaceDisp("make on device", ElementType::mNumDofsPerNode);

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        tComputeSurfaceDispA.setChildNode(iChildNode);

        Plato::ScalarMultiVector tSurfaceDispA("make on device", tPair.surfaceA.childElements().size(), ElementType::mNumDofsPerNode);
        tComputeSurfaceDispA(tPair.surfaceA.childElements(), tDispWS, tSurfaceDispA);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispA, tChildCellOrdinal, Kokkos::ALL()) );
        for(int iDof=0; iDof<tSurfaceDisp_Gold[iChildNode].size(); iDof++){
            TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iChildNode][iDof], 1e-12);
        }
    }

    tSurfaceDisp_Gold = {
        {0.0022 / 3.0, 0.0023 / 3.0, 0.0024 / 3.0},
        {0.0016 / 3.0, 0.0017 / 3.0, 0.0018 / 3.0},
        {0.0019 / 3.0, 0.0020 / 3.0, 0.0021 / 3.0}
    };

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        tComputeSurfaceDispB.setChildNode(iChildNode);

        Plato::ScalarMultiVector tSurfaceDispB("make on device", tPair.surfaceB.childElements().size(), ElementType::mNumDofsPerNode);
        tComputeSurfaceDispB(tPair.surfaceB.childElements(), tDispWS, tSurfaceDispB);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispB, tChildCellOrdinal, Kokkos::ALL()) );
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
        tComputeSurfaceDispA.setChildNode(iChildNode);

        Plato::ScalarMultiVector tSurfaceDispA("make on device", tPair.surfaceA.childElements().size(), ElementType::mNumDofsPerNode);
        tComputeSurfaceDispA(tPair.surfaceA.childElements(), tDispWS, tSurfaceDispA);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispA, tChildCellOrdinal, Kokkos::ALL()) );
        for(int iDof=0; iDof<tSurfaceDisp_Gold[iChildNode].size(); iDof++){
            TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iChildNode][iDof], 1e-12);
        }
    }

    tSurfaceDisp_Gold = {
        {0.0022 / 3.0, 0.0023 / 3.0, 0.0024 / 3.0},
        {0.0019 / 3.0, 0.0020 / 3.0, 0.0021 / 3.0},
        {0.0001 / 3.0, 0.0002 / 3.0, 0.0003 / 3.0}
    };

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        tComputeSurfaceDispB.setChildNode(iChildNode);

        Plato::ScalarMultiVector tSurfaceDispB("make on device", tPair.surfaceB.childElements().size(), ElementType::mNumDofsPerNode);
        tComputeSurfaceDispB(tPair.surfaceB.childElements(), tDispWS, tSurfaceDispB);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispB, tChildCellOrdinal, Kokkos::ALL()) );
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

    check_element_type_is_tet(tMesh);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
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

    // get contact pair info
    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);
    auto tPair = tPairs[0]; // there is only 1 pair

    // construct compute surface displacement functors for side A
    Plato::Contact::SurfaceDisplacement<EvaluationType> computeChildSurfaceDispA(tPair.surfaceA.childFaceLocalNodes(), -1.0);
    Plato::Contact::ProjectedSurfaceDisplacement<EvaluationType> 
        computeParentSurfaceDispA(tPair.surfaceA.parentElements(), tPair.surfaceA.mappedChildNodeLocations(), tPair.surfaceA.elementWiseChildMap(), tMesh);

    // construct compute surface displacement functors for side B
    Plato::Contact::SurfaceDisplacement<EvaluationType> computeChildSurfaceDispB(tPair.surfaceB.childFaceLocalNodes());
    Plato::Contact::ProjectedSurfaceDisplacement<EvaluationType> 
        computeParentSurfaceDispB(tPair.surfaceB.parentElements(), tPair.surfaceB.mappedChildNodeLocations(), tPair.surfaceB.elementWiseChildMap(), tMesh, -1.0);

    // test computation of displacement difference (dummy contact force) for side A
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultA("dummy contact force", tPair.surfaceA.childElements().size(), ElementType::mNumDofsPerCell);
    tResidual.dummy_contact_force(tSpatialModel,tPair.surfaceA.childSideSet(),tDispWS,computeChildSurfaceDispA,tResultA); // child face contributions

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        computeParentSurfaceDispA.setChildNode(iChildNode);
        tResidual.dummy_contact_force(tSpatialModel,tPair.surfaceA.childSideSet(),tDispWS,computeParentSurfaceDispA,tResultA); // parent face contributions
    }

    std::vector<std::vector<double>> tResult_Gold = {
        {0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0022 / 3.0, 0.0, 0.0, 0.0},
        {0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0, 0.0, 0.0}
    };

    auto tResult_Host = Plato::TestHelpers::get( tResultA );

    for(int iCell=0; iCell<int(tPair.surfaceA.childElements().size()); iCell++){
        for(int iDof=0; iDof<ElementType::mNumNodesPerFace*ElementType::mNumDofsPerNode; iDof++){
            TEST_FLOATING_EQUALITY(tResult_Host(iCell,iDof), tResult_Gold[iCell][iDof], 1e-12);
      }
    }

    // test computation of displacement difference (dummy contact force) for side B
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultB("dummy contact force", tPair.surfaceB.childElements().size(), ElementType::mNumDofsPerCell);
    tResidual.dummy_contact_force(tSpatialModel,tPair.surfaceB.childSideSet(),tDispWS,computeChildSurfaceDispB,tResultB); // child face contributions

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        computeParentSurfaceDispB.setChildNode(iChildNode);
        tResidual.dummy_contact_force(tSpatialModel,tPair.surfaceB.childSideSet(),tDispWS,computeParentSurfaceDispB,tResultB); // parent face contributions
    }

    tResult_Gold = {
        {0.0, 0.0, 0.0, 0.0012 / 3.0, 0.0012 / 3.0, 0.0012 / 3.0, 0.0012 / 3.0, 0.0012 / 3.0, 0.0012 / 3.0, 0.0012 / 3.0, 0.0012 / 3.0, 0.0012 / 3.0},
        {0.0, 0.0, 0.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0, 0.0019 / 3.0}
    };

    tResult_Host = Plato::TestHelpers::get( tResultB );

    for(int iCell=0; iCell<int(tPair.surfaceB.childElements().size()); iCell++){
        for(int iDof=0; iDof<ElementType::mNumNodesPerFace*ElementType::mNumDofsPerNode; iDof++){
            TEST_FLOATING_EQUALITY(tResult_Host(iCell,iDof), tResult_Gold[iCell][iDof], 1e-12);
      }
    }

}

}