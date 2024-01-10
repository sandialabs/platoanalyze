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
#include "PlatoMathHelpers.hpp"

#include "WorksetBase.hpp"
#include "SpatialModel.hpp"

#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/VectorFunction.hpp"

#include "Mechanics.hpp"

#include "contact/ContactPair.hpp"
#include "contact/ContactUtils.hpp"
#include "contact/SurfaceDisplacementFactory.hpp"
#include "contact/ContactForceFactory.hpp"

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
    (const Plato::SpatialModel                                                       & aSpatialModel,
     const std::string                                                               & aSideSet,
     const Plato::ScalarMultiVectorT<StateScalarType>                                & aState,
           Teuchos::RCP<Plato::Contact::AbstractSurfaceDisplacement<EvaluationType>>   aComputeSurfaceDisp,
           Plato::ScalarMultiVectorT<ResultScalarType>                               & aResult)
    {
        auto tElementOrds   = aSpatialModel.Mesh->GetSideSetElements(aSideSet);
        Plato::OrdinalType tNumFaces = tElementOrds.size();

        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tCubatureWeights = ElementType::Face::getCubWeights();
        auto tNumPoints = tCubatureWeights.size();

        Plato::ScalarArray3DT<ResultScalarType> tSurfaceDisplacement("", tNumFaces, tNumPoints, ElementType::mNumDofsPerNode);
        (*aComputeSurfaceDisp)(tElementOrds, aState, tSurfaceDisplacement);

        auto tLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(aSideSet);

        Kokkos::parallel_for("contact force", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
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
                    ResultScalarType tResult = tBasisValues(tNode)*tSurfaceDisplacement(iCellOrdinal, iGPOrdinal, tDof);
                    Kokkos::atomic_add(&aResult(iCellOrdinal, tElementDofOrdinal), tResult);
                }
            }

        });
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
        "        <Parameter name='Search Tolerance' type='double' value='0.5' />  \n"
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

    // test search tolerance
    TEST_FLOATING_EQUALITY(tPair.searchTolerance, 0.5, 1e-13)
}

TEUCHOS_UNIT_TEST(UtilsTests, ParseAllContactPairs)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);

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

    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
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

TEUCHOS_UNIT_TEST(UtilsTests, CheckForMissingParentElements)
{
    std::vector<Plato::OrdinalType> tParentElements = {1, 5, -2, 7};
    auto dParentElements = Plato::TestHelpers::create_device_view(tParentElements);

    TEST_THROW(Plato::Contact::check_for_missing_parent_elements(dParentElements), std::runtime_error);
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

    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
    auto tPair = tPairs[0];

    TEST_THROW(tPair.surfaceA.parentElements(), std::runtime_error);
    TEST_THROW(tPair.surfaceA.elementWiseChildMap(), std::runtime_error);
    TEST_THROW(tPair.surfaceA.mappedChildNodeLocations(), std::runtime_error);

    TEST_THROW(tPair.surfaceB.parentElements(), std::runtime_error);
    TEST_THROW(tPair.surfaceB.elementWiseChildMap(), std::runtime_error);
    TEST_THROW(tPair.surfaceB.mappedChildNodeLocations(), std::runtime_error);
}

TEUCHOS_UNIT_TEST(FunctorTests, ComputeContactForce_CompliantContactForce)
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
    Plato::OrdinalType tNumPoints = ElementType::Face::getCubWeights().size();
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;

    auto tPairsParams = tContactParams->sublist("Pairs");
    const auto& tMyName = tPairsParams.name(tPairsParams.begin());
    Teuchos::ParameterList& tPairParams = tPairsParams.sublist(tMyName);

    // get pair data
    Plato::Contact::ContactPair tPair = Plato::Contact::parse_contact_pair(tPairParams, tMesh);
    auto tChildElements = tPair.surfaceA.childElements();
    Plato::OrdinalType tNumChildElements = tChildElements.size();
    auto tChildFaceLocalNodes = tPair.surfaceA.childFaceLocalNodes();

    // apply contact penalty
    Plato::Contact::ContactForceFactory<EvaluationType> tFactory;
    auto computeContactForce = tFactory.create(tPair.penaltyType, tPair.penaltyValue);

    std::vector<Plato::Scalar> tProjectedDisp = {45.3, 66.54, 77.88};
    auto dProjectedDisp = Plato::TestHelpers::create_device_view(tProjectedDisp);
    Plato::ScalarArray3D tFullProjectedDisp("",tNumChildElements,tNumPoints,ElementType::mNumSpatialDims);
    Kokkos::parallel_for("fill in for device", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumChildElements, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
    {
        for(Plato::OrdinalType iDim = 0; iDim < ElementType::mNumSpatialDims; iDim++)
            tFullProjectedDisp(iCellOrdinal,iGPOrdinal,iDim) = dProjectedDisp(iDim);
    });

    Plato::ScalarArray3D tPenalizedDisp("",tNumChildElements,tNumPoints,ElementType::mNumSpatialDims);
    Plato::ScalarArray3D tConfig("Dummy Config Workset", tMesh->NumElements(), ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims);
    (*computeContactForce)(tChildElements, tChildFaceLocalNodes, tFullProjectedDisp, tConfig, tPenalizedDisp);

    // test
    std::vector<Plato::Scalar> tPenalizedDisp_Gold = {45.3e5, 66.54e5, 77.88e5};
    auto tPenalizedDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tPenalizedDisp, 0, 0, Kokkos::ALL()) );

    for(int iOrd=0; iOrd<tPenalizedDisp_Gold.size(); iOrd++)
        TEST_FLOATING_EQUALITY(tPenalizedDisp_Host(iOrd), tPenalizedDisp_Gold[iOrd], 1.0e-13);
}

TEUCHOS_UNIT_TEST(FunctorTests, ComputeContactForce_NormalContactForce)
{
    Teuchos::RCP<Teuchos::ParameterList> tContactParams =
        Teuchos::getParametersFromXmlString(
        "  <ParameterList name='Contact'>                                                     \n"
        "    <ParameterList name='Pairs'>                                                     \n"
        "      <ParameterList name='Pair 1'>                                                  \n"
        "        <Parameter name='Initial Gap' type='Array(double)' value='{1.0,0.0,0.0}' />  \n"
        "        <Parameter name='Penalty Type' type='string' value='normal' />  \n"
        "        <Parameter name='Penalty Value' type='double' value='1.0e5' />  \n"
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
    Plato::OrdinalType tNumPoints = ElementType::Face::getCubWeights().size();
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;

    // get config workset
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("Config Workset", tMesh->NumElements(), ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims);
    tWorksetBase.worksetConfig(tConfigWS);

    // parse pair input
    auto tPairsParams = tContactParams->sublist("Pairs");
    const auto& tMyName = tPairsParams.name(tPairsParams.begin());
    Teuchos::ParameterList& tPairParams = tPairsParams.sublist(tMyName);

    // get pair data
    Plato::Contact::ContactPair tPair = Plato::Contact::parse_contact_pair(tPairParams, tMesh);
    auto tChildElements = tPair.surfaceA.childElements();
    Plato::OrdinalType tNumChildElements = tChildElements.size();
    auto tChildFaceLocalNodes = tPair.surfaceA.childFaceLocalNodes();

    // apply contact penalty
    Plato::Contact::ContactForceFactory<EvaluationType> tFactory;
    auto computeContactForce = tFactory.create(tPair.penaltyType, tPair.penaltyValue);

    std::vector<Plato::Scalar> tProjectedDisp = {45.3, 66.54, 77.88};
    auto dProjectedDisp = Plato::TestHelpers::create_device_view(tProjectedDisp);
    Plato::ScalarArray3D tFullProjectedDisp("",tNumChildElements,tNumPoints,ElementType::mNumSpatialDims);
    Kokkos::parallel_for("fill in for device", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumChildElements, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
    {
        for(Plato::OrdinalType iDim = 0; iDim < ElementType::mNumSpatialDims; iDim++)
            tFullProjectedDisp(iCellOrdinal,iGPOrdinal,iDim) = dProjectedDisp(iDim);
    });

    Plato::ScalarArray3D tPenalizedDisp("",tNumChildElements,tNumPoints,ElementType::mNumSpatialDims);
    (*computeContactForce)(tChildElements, tChildFaceLocalNodes, tFullProjectedDisp, tConfigWS, tPenalizedDisp);

    // test
    std::vector<Plato::Scalar> tPenalizedDisp_Gold = {45.3e5, 0, 0};
    auto tPenalizedDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tPenalizedDisp, 0, 0, Kokkos::ALL()) );

    for(int iOrd=0; iOrd<tPenalizedDisp_Gold.size(); iOrd++)
        TEST_FLOATING_EQUALITY(tPenalizedDisp_Host(iOrd), tPenalizedDisp_Gold[iOrd], 1.0e-13);
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_ChildElementContrbution)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    check_element_type_is_tet(tMesh);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tNumPoints = tCubatureWeights.size();

    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);
     
    // get contact pair info
    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
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
    Plato::Contact::SurfaceDisplacementFactory<EvaluationType> tFactory;
    auto tComputeSurfaceDispA = tFactory.createChildContribution(tPair.surfaceA, -1.0);
    auto tComputeSurfaceDispB = tFactory.createChildContribution(tPair.surfaceB, -1.0);

    // compute surface displacement for all child face cells
    Plato::ScalarArray3D tSurfaceDispA("make on device", tPair.surfaceA.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
    (*tComputeSurfaceDispA)(tPair.surfaceA.childElements(), tDispWS, tSurfaceDispA);

    Plato::ScalarArray3D tSurfaceDispB("make on device", tPair.surfaceB.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
    (*tComputeSurfaceDispB)(tPair.surfaceB.childElements(), tDispWS, tSurfaceDispB);

    // test surface displacement child face cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;

    auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispA, tChildCellOrdinal, 0, Kokkos::ALL()) );
    std::vector<double> tSurfaceDisp_Gold = {-0.0012, -0.0013, -0.0014};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }

    tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispB, tChildCellOrdinal, 0, Kokkos::ALL()) );
    tSurfaceDisp_Gold = {-0.0031, -0.0032, -0.0033};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }

    // test surface displacement child face cell 1
    tChildCellOrdinal = 1;

    tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispA, tChildCellOrdinal, 0, Kokkos::ALL()) );
    tSurfaceDisp_Gold = {-0.0013, -0.0014, -0.0015};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }

    tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispB, tChildCellOrdinal, 0, Kokkos::ALL()) );
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
    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tNumPoints = tCubatureWeights.size();
    
    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // get contact pair info
    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);
    auto tPair = tPairs[0]; // there is only 1 pair

    // construct compute surface displacement functors
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    Plato::Contact::SurfaceDisplacementFactory<EvaluationType> tFactory;
    auto tComputeSurfaceDispA = tFactory.createParentContribution(tPair.surfaceA, tMesh);
    auto tComputeSurfaceDispB = tFactory.createParentContribution(tPair.surfaceB, tMesh);

    // test surface displacement terms for each child node on child cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;

    std::vector<std::vector<double>> tSurfaceDisp_Gold = {
        {0.0037 / 3.0, 0.0038 / 3.0, 0.0039 / 3.0},
        {0.0034 / 3.0, 0.0035 / 3.0, 0.0036 / 3.0},
        {0.0031 / 3.0, 0.0032 / 3.0, 0.0033 / 3.0}
    };

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        tComputeSurfaceDispA->setChildNode(iChildNode);

        Plato::ScalarArray3D tSurfaceDispA("make on device", tPair.surfaceA.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
        (*tComputeSurfaceDispA)(tPair.surfaceA.childElements(), tDispWS, tSurfaceDispA);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispA, tChildCellOrdinal, 0, Kokkos::ALL()) );
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
        tComputeSurfaceDispB->setChildNode(iChildNode);

        Plato::ScalarArray3D tSurfaceDispB("make on device", tPair.surfaceB.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
        (*tComputeSurfaceDispB)(tPair.surfaceB.childElements(), tDispWS, tSurfaceDispB);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispB, tChildCellOrdinal, 0, Kokkos::ALL()) );
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
        tComputeSurfaceDispA->setChildNode(iChildNode);

        Plato::ScalarArray3D tSurfaceDispA("make on device", tPair.surfaceA.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
        (*tComputeSurfaceDispA)(tPair.surfaceA.childElements(), tDispWS, tSurfaceDispA);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispA, tChildCellOrdinal, 0, Kokkos::ALL()) );
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
        tComputeSurfaceDispB->setChildNode(iChildNode);

        Plato::ScalarArray3D tSurfaceDispB("make on device", tPair.surfaceB.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
        (*tComputeSurfaceDispB)(tPair.surfaceB.childElements(), tDispWS, tSurfaceDispB);

        auto tSurfaceDisp_Host = Plato::TestHelpers::get( Kokkos::subview(tSurfaceDispB, tChildCellOrdinal, 0, Kokkos::ALL()) );
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
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // construct dummy residual class
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    DummyResidual<EvaluationType> tResidual;

    // get contact pair info
    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);
    auto tPair = tPairs[0]; // there is only 1 pair

    // construct compute surface displacement functors for side A
    Plato::Contact::SurfaceDisplacementFactory<EvaluationType> tFactory;

    auto computeChildSurfaceDispA  = tFactory.createChildContribution(tPair.surfaceA);
    auto computeParentSurfaceDispA = tFactory.createParentContribution(tPair.surfaceA, tMesh, -1.0);

    // construct compute surface displacement functors for side B
    auto computeChildSurfaceDispB  = tFactory.createChildContribution(tPair.surfaceB);
    auto computeParentSurfaceDispB = tFactory.createParentContribution(tPair.surfaceB, tMesh, -1.0);

    // test computation of displacement difference (dummy contact force) for side A
    Plato::ScalarMultiVectorT<Plato::Scalar> tResultA("dummy contact force", tPair.surfaceA.childElements().size(), ElementType::mNumDofsPerCell);
    tResidual.dummy_contact_force(tSpatialModel,tPair.surfaceA.childSideSet(),tDispWS,computeChildSurfaceDispA,tResultA); // child face contributions

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        computeParentSurfaceDispA->setChildNode(iChildNode);
        tResidual.dummy_contact_force(tSpatialModel,tPair.surfaceA.childSideSet(),tDispWS,computeParentSurfaceDispA,tResultA); // parent face contributions
    }

    std::vector<std::vector<double>> tResult_Gold = {
        {-0.0022 / 3.0, -0.0022 / 3.0, -0.0022 / 3.0, -0.0022 / 3.0, -0.0022 / 3.0, -0.0022 / 3.0, -0.0022 / 3.0, -0.0022 / 3.0, -0.0022 / 3.0, 0.0, 0.0, 0.0},
        {-0.0019 / 3.0, -0.0019 / 3.0, -0.0019 / 3.0, -0.0019 / 3.0, -0.0019 / 3.0, -0.0019 / 3.0, -0.0019 / 3.0, -0.0019 / 3.0, -0.0019 / 3.0, 0.0, 0.0, 0.0}
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
        computeParentSurfaceDispB->setChildNode(iChildNode);
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

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_ChildElementJacobian)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    check_element_type_is_tet(tMesh);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tNumPoints = tCubatureWeights.size();

    // set evaluation type to jacobian
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using StateScalar    = typename EvaluationType::StateScalarType;

    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<StateScalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);
     
    // get contact pair info
    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
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
    Plato::Contact::SurfaceDisplacementFactory<EvaluationType> tFactory;
    auto tComputeSurfaceDispA = tFactory.createChildContribution(tPair.surfaceA, -1.0);
    auto tComputeSurfaceDispB = tFactory.createChildContribution(tPair.surfaceB, -1.0);

    // compute surface displacement for all child face cells
    Plato::ScalarArray3DT<StateScalar> tSurfaceDispA("make on device", tPair.surfaceA.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
    (*tComputeSurfaceDispA)(tPair.surfaceA.childElements(), tDispWS, tSurfaceDispA);

    Plato::ScalarArray3DT<StateScalar> tSurfaceDispB("make on device", tPair.surfaceB.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
    (*tComputeSurfaceDispB)(tPair.surfaceB.childElements(), tDispWS, tSurfaceDispB);

    // test surface A displacement jacobian derivatives for child face cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;
    Plato::ScalarVector tADerivative0("", ElementType::mNumDofsPerCell);
    Plato::ScalarVector tADerivative1("", ElementType::mNumDofsPerCell);
    Plato::ScalarVector tADerivative2("", ElementType::mNumDofsPerCell);

    Kokkos::parallel_for("get derivatives for testing", Kokkos::RangePolicy<Plato::OrdinalType>(0,ElementType::mNumDofsPerCell), KOKKOS_LAMBDA(Plato::OrdinalType iOrd)
    {
        tADerivative0(iOrd) = tSurfaceDispA(tChildCellOrdinal, 0, 0).dx(iOrd);
        tADerivative1(iOrd) = tSurfaceDispA(tChildCellOrdinal, 0, 1).dx(iOrd);
        tADerivative2(iOrd) = tSurfaceDispA(tChildCellOrdinal, 0, 2).dx(iOrd);
    });

    auto tADerivative0_Host = Plato::TestHelpers::get( tADerivative0 );
    auto tADerivative1_Host = Plato::TestHelpers::get( tADerivative1 );
    auto tADerivative2_Host = Plato::TestHelpers::get( tADerivative2 );

    std::vector<double> tADerivative0_Gold = {-1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> tADerivative1_Gold = {0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, 0.0, 0.0};
    std::vector<double> tADerivative2_Gold = {0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, 0.0};

    for(int iDof=0; iDof<tADerivative0_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tADerivative0_Host(iDof), tADerivative0_Gold[iDof], 1e-12);
        TEST_FLOATING_EQUALITY(tADerivative1_Host(iDof), tADerivative1_Gold[iDof], 1e-12);
        TEST_FLOATING_EQUALITY(tADerivative2_Host(iDof), tADerivative2_Gold[iDof], 1e-12);
    }

    // test surface B displacement jacobian derivatives for child face cell 0
    tChildCellOrdinal = 0;
    Plato::ScalarVector tBDerivative0("", ElementType::mNumDofsPerCell);
    Plato::ScalarVector tBDerivative1("", ElementType::mNumDofsPerCell);
    Plato::ScalarVector tBDerivative2("", ElementType::mNumDofsPerCell);

    Kokkos::parallel_for("get derivatives for testing", Kokkos::RangePolicy<Plato::OrdinalType>(0,ElementType::mNumDofsPerCell), KOKKOS_LAMBDA(Plato::OrdinalType iOrd)
    {
        tBDerivative0(iOrd) = tSurfaceDispB(tChildCellOrdinal, 0, 0).dx(iOrd);
        tBDerivative1(iOrd) = tSurfaceDispB(tChildCellOrdinal, 0, 1).dx(iOrd);
        tBDerivative2(iOrd) = tSurfaceDispB(tChildCellOrdinal, 0, 2).dx(iOrd);
    });

    auto tBDerivative0_Host = Plato::TestHelpers::get( tBDerivative0 );
    auto tBDerivative1_Host = Plato::TestHelpers::get( tBDerivative1 );
    auto tBDerivative2_Host = Plato::TestHelpers::get( tBDerivative2 );

    std::vector<double> tBDerivative0_Gold = {0.0, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0};
    std::vector<double> tBDerivative1_Gold = {0.0, 0.0, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0};
    std::vector<double> tBDerivative2_Gold = {0.0, 0.0, 0.0, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3, 0.0, 0.0, -1.0 / 3};

    for(int iDof=0; iDof<tBDerivative0_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tBDerivative0_Host(iDof), tBDerivative0_Gold[iDof], 1e-12);
        TEST_FLOATING_EQUALITY(tBDerivative1_Host(iDof), tBDerivative1_Gold[iDof], 1e-12);
        TEST_FLOATING_EQUALITY(tBDerivative2_Host(iDof), tBDerivative2_Gold[iDof], 1e-12);
    }
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_SingleParentElementJacobian)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    check_element_type_is_tet(tMesh);
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tNumPoints = tCubatureWeights.size();

    // set evaluation type to jacobian
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using StateScalar    = typename EvaluationType::StateScalarType;
    
    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<StateScalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // get contact pair info
    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);
    auto tPair = tPairs[0]; // there is only 1 pair

    // construct compute surface displacement functors
    Plato::Contact::SurfaceDisplacementFactory<EvaluationType> tFactory;
    auto tComputeSurfaceDispA = tFactory.createParentContribution(tPair.surfaceA, tMesh);
    auto tComputeSurfaceDispB = tFactory.createParentContribution(tPair.surfaceB, tMesh);

    // test surface A displacement terms for each child node on child cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;

    std::vector<std::vector<double>> tADerivative0_Gold = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0}
    };

    std::vector<std::vector<double>> tADerivative1_Gold = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0}
    };

    std::vector<std::vector<double>> tADerivative2_Gold = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0}
    };

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        tComputeSurfaceDispA->setChildNode(iChildNode);

        Plato::ScalarArray3DT<StateScalar> tSurfaceDispA("make on device", tPair.surfaceA.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
        (*tComputeSurfaceDispA)(tPair.surfaceA.childElements(), tDispWS, tSurfaceDispA);

        Plato::ScalarVector tADerivative0("", ElementType::mNumDofsPerCell);
        Plato::ScalarVector tADerivative1("", ElementType::mNumDofsPerCell);
        Plato::ScalarVector tADerivative2("", ElementType::mNumDofsPerCell);
        Kokkos::parallel_for("get derivatives for testing", Kokkos::RangePolicy<Plato::OrdinalType>(0,ElementType::mNumDofsPerCell), KOKKOS_LAMBDA(Plato::OrdinalType iOrd)
        {
            tADerivative0(iOrd) = tSurfaceDispA(tChildCellOrdinal, 0, 0).dx(iOrd);
            tADerivative1(iOrd) = tSurfaceDispA(tChildCellOrdinal, 0, 1).dx(iOrd);
            tADerivative2(iOrd) = tSurfaceDispA(tChildCellOrdinal, 0, 2).dx(iOrd);
        });

        auto tADerivative0_Host = Plato::TestHelpers::get( tADerivative0 );
        auto tADerivative1_Host = Plato::TestHelpers::get( tADerivative1 );
        auto tADerivative2_Host = Plato::TestHelpers::get( tADerivative2 );
        for(int iDof=0; iDof<tADerivative0_Gold[iChildNode].size(); iDof++){
            TEST_FLOATING_EQUALITY(tADerivative0_Host(iDof), tADerivative0_Gold[iChildNode][iDof], 1e-12);
            TEST_FLOATING_EQUALITY(tADerivative1_Host(iDof), tADerivative1_Gold[iChildNode][iDof], 1e-12);
            TEST_FLOATING_EQUALITY(tADerivative2_Host(iDof), tADerivative2_Gold[iChildNode][iDof], 1e-12);
        }
    }

    // test surface B displacement terms for each child node on child cell 0
    std::vector<std::vector<double>> tBDerivative0_Gold = {
        {0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0},
    };

    std::vector<std::vector<double>> tBDerivative1_Gold = {
        {0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0},
    };

    std::vector<std::vector<double>> tBDerivative2_Gold = {
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / 3, 0.0, 0.0, 0.0},
    };

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        tComputeSurfaceDispB->setChildNode(iChildNode);

        Plato::ScalarArray3DT<StateScalar> tSurfaceDispB("make on device", tPair.surfaceB.childElements().size(), tNumPoints, ElementType::mNumDofsPerNode);
        (*tComputeSurfaceDispB)(tPair.surfaceB.childElements(), tDispWS, tSurfaceDispB);

        Plato::ScalarVector tBDerivative0("", ElementType::mNumDofsPerCell);
        Plato::ScalarVector tBDerivative1("", ElementType::mNumDofsPerCell);
        Plato::ScalarVector tBDerivative2("", ElementType::mNumDofsPerCell);
        Kokkos::parallel_for("get derivatives for testing", Kokkos::RangePolicy<Plato::OrdinalType>(0,ElementType::mNumDofsPerCell), KOKKOS_LAMBDA(Plato::OrdinalType iOrd)
        {
            tBDerivative0(iOrd) = tSurfaceDispB(tChildCellOrdinal, 0, 0).dx(iOrd);
            tBDerivative1(iOrd) = tSurfaceDispB(tChildCellOrdinal, 0, 1).dx(iOrd);
            tBDerivative2(iOrd) = tSurfaceDispB(tChildCellOrdinal, 0, 2).dx(iOrd);
        });

        auto tBDerivative0_Host = Plato::TestHelpers::get( tBDerivative0 );
        auto tBDerivative1_Host = Plato::TestHelpers::get( tBDerivative1 );
        auto tBDerivative2_Host = Plato::TestHelpers::get( tBDerivative2 );
        for(int iDof=0; iDof<tBDerivative0_Gold[iChildNode].size(); iDof++){
            TEST_FLOATING_EQUALITY(tBDerivative0_Host(iDof), tBDerivative0_Gold[iChildNode][iDof], 1e-12);
            TEST_FLOATING_EQUALITY(tBDerivative1_Host(iDof), tBDerivative1_Gold[iChildNode][iDof], 1e-12);
            TEST_FLOATING_EQUALITY(tBDerivative2_Host(iDof), tBDerivative2_Gold[iChildNode][iDof], 1e-12);
        }
    }
}

TEUCHOS_UNIT_TEST(ResidualTests, ElastoStatic_NoBodyContribution)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                           \n"
        "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
        "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
        "  <ParameterList name='Elliptic'>                                                \n"
        "    <ParameterList name='Penalty Function'>                                      \n"
        "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
        "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
        "    </ParameterList>                                                             \n"
        "  </ParameterList>                                                               \n"

        "  <ParameterList name='Spatial Model'>                                         \n"
        "    <ParameterList name='Domains'>                                             \n"
        "      <ParameterList name='Box 1'>                                             \n"
        "        <Parameter name='Element Block' type='string' value='block_1'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Ether'/>   \n"
        "      </ParameterList>                                                         \n"
        "      <ParameterList name='Box 2'>                                             \n"
        "        <Parameter name='Element Block' type='string' value='block_2'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Ether'/>   \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"

        "  <ParameterList name='Contact'>                                                     \n"
        "    <ParameterList name='Pairs'>                                                     \n"
        "      <ParameterList name='Pair 1'>                                                  \n"
        "        <Parameter name='Initial Gap' type='Array(double)' value='{1.0,0.0,0.0}' />  \n"
        "        <Parameter name='Penalty Value' type='Array(double)' value='{1.0e4,1.0e4,1.0e4}' />  \n"
        "        <Parameter name='Penalty Type' type='string' value='tensor' />  \n"
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
        "    <ParameterList name='Ether'>                                         \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                          \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.0'/>         \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='0.0'/>       \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "</ParameterList>                                                               \n"
    );

    // setup spatial model
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    // add contact to spatial model
    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);

    tSpatialModel.addContact(tPairs);

    // create dummy control vector (all 1s)
    std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
    auto z = Plato::TestHelpers::create_device_view(z_host);

    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    // compute and test residual
    Plato::Elliptic::VectorFunction<::Plato::Mechanics<Plato::Tet4>>
        tVectorFunction(tSpatialModel, tDataMap, *tInputs, tInputs->get<std::string>("PDE Constraint"));

    auto tResidual = tVectorFunction.value(u,z);

    auto tResidual_Host = Plato::TestHelpers::get( tResidual );

    // 1/3 is the face basis function value at gauss point (for tet4)
    // 1/2 is the face weight at gauss point (for tet4)
    std::vector<Plato::Scalar> tResidual_Gold = {
        -0.0041e4 / 3 / 2, -0.0041e4 / 3 / 2, -0.0041e4 / 3 / 2,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        -0.0041e4 / 3 / 2, -0.0041e4 / 3 / 2, -0.0041e4 / 3 / 2,
        -0.0022e4 / 3 / 2, -0.0022e4 / 3 / 2, -0.0022e4 / 3 / 2,
        -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2,

        0.0, 0.0, 0.0,
        0.0031e4 / 3 / 2, 0.0031e4 / 3 / 2, 0.0031e4 / 3 / 2,
        0.0012e4 / 3 / 2, 0.0012e4 / 3 / 2, 0.0012e4 / 3 / 2,
        0.0031e4 / 3 / 2, 0.0031e4 / 3 / 2, 0.0031e4 / 3 / 2,
        0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };

    for(int iVal=0; iVal<tResidual_Gold.size(); iVal++){
        TEST_FLOATING_EQUALITY(tResidual_Host(iVal), tResidual_Gold[iVal], 1e-12);
    }
}

TEUCHOS_UNIT_TEST(JacobianTests, ElastoStatic_NoBodyContribution)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                           \n"
        "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
        "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
        "  <ParameterList name='Elliptic'>                                                \n"
        "    <ParameterList name='Penalty Function'>                                      \n"
        "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
        "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
        "    </ParameterList>                                                             \n"
        "  </ParameterList>                                                               \n"

        "  <ParameterList name='Spatial Model'>                                         \n"
        "    <ParameterList name='Domains'>                                             \n"
        "      <ParameterList name='Box 1'>                                             \n"
        "        <Parameter name='Element Block' type='string' value='block_1'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Ether'/>   \n"
        "      </ParameterList>                                                         \n"
        "      <ParameterList name='Box 2'>                                             \n"
        "        <Parameter name='Element Block' type='string' value='block_2'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Ether'/>   \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"

        "  <ParameterList name='Contact'>                                                     \n"
        "    <ParameterList name='Pairs'>                                                     \n"
        "      <ParameterList name='Pair 1'>                                                  \n"
        "        <Parameter name='Initial Gap' type='Array(double)' value='{1.0,0.0,0.0}' />  \n"
        "        <Parameter name='Penalty Value' type='Array(double)' value='{1.0e4,1.0e4,1.0e4}' />  \n"
        "        <Parameter name='Penalty Type' type='string' value='tensor' />  \n"
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
        "    <ParameterList name='Ether'>                                         \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                          \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.0'/>         \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='0.0'/>       \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "</ParameterList>                                                               \n"
    );

    // setup spatial model
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    // add contact to spatial model
    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);

    tSpatialModel.addContact(tPairs);

    // create dummy control vector (all 1s)
    std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
    auto z = Plato::TestHelpers::create_device_view(z_host);

    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    // compute and test jacobian
    Plato::Elliptic::VectorFunction<::Plato::Mechanics<Plato::Tet4>>
        tVectorFunction(tSpatialModel, tDataMap, *tInputs, tInputs->get<std::string>("PDE Constraint"));

    auto tJacobian = tVectorFunction.gradient_u(u,z);
    auto tEntries = tJacobian->entries();

    auto tEntries_Host = Plato::TestHelpers::get( tEntries );

    // 1/3 is the face basis function value at gauss point (for tet4)
    // 1/2 is the face weight at gauss point (for tet4)
    std::vector<Plato::Scalar> tEntries_Gold = { 
        2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
 
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2,

        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,

        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,

        



        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2,
        -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2,
        -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2, 0, 0, 0, -2.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2, 0, 0, 0, 2.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2, 0, 0, 0, -1.0e4 / 9 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,
        1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2, 0, 0, 0, 1.0e4 / 9 / 2,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        };

    for(int iVal=0; iVal<tEntries_Gold.size(); iVal++){
        TEST_FLOATING_EQUALITY(tEntries_Host(iVal), tEntries_Gold[iVal], 1e-12);
    }
}

TEUCHOS_UNIT_TEST(GradientXTests, ElastoStatic_NoBodyContribution)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                           \n"
        "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
        "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
        "  <ParameterList name='Elliptic'>                                                \n"
        "    <ParameterList name='Penalty Function'>                                      \n"
        "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
        "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
        "    </ParameterList>                                                             \n"
        "  </ParameterList>                                                               \n"

        "  <ParameterList name='Spatial Model'>                                         \n"
        "    <ParameterList name='Domains'>                                             \n"
        "      <ParameterList name='Box 1'>                                             \n"
        "        <Parameter name='Element Block' type='string' value='block_1'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Ether'/>   \n"
        "      </ParameterList>                                                         \n"
        "      <ParameterList name='Box 2'>                                             \n"
        "        <Parameter name='Element Block' type='string' value='block_2'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Ether'/>   \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"

        "  <ParameterList name='Contact'>                                                     \n"
        "    <ParameterList name='Pairs'>                                                     \n"
        "      <ParameterList name='Pair 1'>                                                  \n"
        "        <Parameter name='Initial Gap' type='Array(double)' value='{1.0,0.0,0.0}' />  \n"
        "        <Parameter name='Penalty Value' type='Array(double)' value='{1.0e4,1.0e4,1.0e4}' />  \n"
        "        <Parameter name='Penalty Type' type='string' value='tensor' />  \n"
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
        "    <ParameterList name='Ether'>                                         \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                          \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.0'/>         \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='0.0'/>       \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "</ParameterList>                                                               \n"
    );

    // setup spatial model
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    // add contact to spatial model
    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), tMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);

    tSpatialModel.addContact(tPairs);

    // create dummy control vector (all 1s)
    std::vector<Plato::Scalar> z_host( tMesh->NumNodes(), 1.0 );
    auto z = Plato::TestHelpers::create_device_view(z_host);

    // create dummy displacement workset from box mesh
    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    // compute and test gradientX
    Plato::Elliptic::VectorFunction<::Plato::Mechanics<Plato::Tet4>>
        tVectorFunction(tSpatialModel, tDataMap, *tInputs, tInputs->get<std::string>("PDE Constraint"));

    auto tGradientXTranspose = tVectorFunction.gradient_x(u,z); // recall this returns (dR/dX)^T

    // get dR/dX from transpose
    auto tNumRows = tGradientXTranspose->numCols();
    auto tNumCols = tGradientXTranspose->numRows();
    auto tNumRowsPerBlock = tGradientXTranspose->numColsPerBlock();
    auto tNumColsPerBlock = tGradientXTranspose->numRowsPerBlock();
    auto tGradientX = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock ) );
    Plato::MatrixTranspose(tGradientXTranspose, tGradientX);

    auto tEntries = tGradientX->entries();

    auto tEntries_Host = Plato::TestHelpers::get( tEntries );

    // 1/3 is the face basis function value at gauss point (for tet4)
    // 1/2 is the face weight at gauss point (for tet4)
    // Surface Area gradients for nodes on element faces:
        // Element 2
            // Node 0: [0 0 1]
            // Node 5: [0 1 0]
            // Node 6: [0 -1 -1]
        // Element 4
            // Node 0: [0 -1 0]
            // Node 5: [0 0 -1]
            // Node 7: [0 1 1]
        // Element 6
            // Node 9:  [0 0 1]
            // Node 10: [0 1 -1]
            // Node 11: [0 -1 0]
        // Element 4
            // Node 9:  [0 1 0]
            // Node 11: [0 0 -1]
            // Node 12: [0 -1 1]
    std::vector<Plato::Scalar> tEntries_Gold = { 
        0, 0.0019e4 / 3 / 2, -0.0022e4 / 3 / 2, 0, 0.0019e4 / 3 / 2, -0.0022e4 / 3 / 2, 0, 0.0019e4 / 3 / 2, -0.0022e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -0.0022e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0022e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0022e4 / 3 / 2, 0.0019e4 / 3 / 2,
        0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2, 0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2, 0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2,
        0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
 
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0.0019e4 / 3 / 2, -0.0022e4 / 3 / 2, 0, 0.0019e4 / 3 / 2, -0.0022e4 / 3 / 2, 0, 0.0019e4 / 3 / 2, -0.0022e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -0.0022e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0022e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0022e4 / 3 / 2, 0.0019e4 / 3 / 2,
        0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2, 0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2, 0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2,
        0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, -0.0022e4 / 3 / 2, 0, 0, -0.0022e4 / 3 / 2, 0, 0, -0.0022e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, -0.0022e4 / 3 / 2, 0, 0, -0.0022e4 / 3 / 2, 0, 0, -0.0022e4 / 3 / 2, 0,
        0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2, 0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2, 0, 0.0022e4 / 3 / 2, 0.0022e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0.0019e4 / 3 / 2, 0, 0, 0.0019e4 / 3 / 2, 0, 0, 0.0019e4 / 3 / 2, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0.0019e4 / 3 / 2, 0, 0, 0.0019e4 / 3 / 2, 0, 0, 0.0019e4 / 3 / 2,
        0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, -0.0019e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        



        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.0019e4 / 3 / 2, 0.0012e4 / 3 / 2, 0, 0.0019e4 / 3 / 2, 0.0012e4 / 3 / 2, 0, 0.0019e4 / 3 / 2, 0.0012e4 / 3 / 2,
        0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2, 0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2, 0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2,
        0, -0.0012e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0012e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0012e4 / 3 / 2, -0.0019e4 / 3 / 2,
        0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0.0012e4 / 3 / 2, 0, 0, 0.0012e4 / 3 / 2, 0, 0, 0.0012e4 / 3 / 2,
        0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2, 0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2, 0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2,
        0, -0.0012e4 / 3 / 2, 0, 0, -0.0012e4 / 3 / 2, 0, 0, -0.0012e4 / 3 / 2, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.0019e4 / 3 / 2, 0.0012e4 / 3 / 2, 0, 0.0019e4 / 3 / 2, 0.0012e4 / 3 / 2, 0, 0.0019e4 / 3 / 2, 0.0012e4 / 3 / 2,
        0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2, 0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2, 0, 0.0012e4 / 3 / 2, -0.0012e4 / 3 / 2,
        0, -0.0012e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0012e4 / 3 / 2, -0.0019e4 / 3 / 2, 0, -0.0012e4 / 3 / 2, -0.0019e4 / 3 / 2,
        0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0.0019e4 / 3 / 2, 0, 0, 0.0019e4 / 3 / 2, 0, 0, 0.0019e4 / 3 / 2, 0,
        0, 0, -0.0019e4 / 3 / 2, 0, 0, -0.0019e4 / 3 / 2, 0, 0, -0.0019e4 / 3 / 2,
        0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2, 0, -0.0019e4 / 3 / 2, 0.0019e4 / 3 / 2,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        };

    for(int iVal=0; iVal<tEntries_Gold.size(); iVal++){
        TEST_FLOATING_EQUALITY(tEntries_Host(iVal), tEntries_Gold[iVal], 1e-12);
    }
}

}
