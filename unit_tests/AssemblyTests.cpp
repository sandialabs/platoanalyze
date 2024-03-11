#include "util/PlatoTestHelpers.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "PlatoUtilities.hpp"

#include "PlatoStaticsTypes.hpp"
#include "Tet4.hpp"
#include "MechanicsElement.hpp"

#include "EngineMesh.hpp"
#include "SpatialModel.hpp"

#include "elliptic/EvaluationTypes.hpp"

#include "WorksetBase.hpp"
#include "ImplicitFunctors.hpp"
#include "MatrixGraphUtils.hpp"

#include "InterpolateFromNodal.hpp"

#ifdef PLATO_MESHMAP
#include "contact/ContactUtils.hpp"
#endif

namespace AssemblyTests
{

template <typename EvaluationType>
class DummyResidual
{
private:
    using ElementType      = typename EvaluationType::ElementType;
    using StateScalarType  = typename EvaluationType::StateScalarType;  
    using ResultScalarType = typename EvaluationType::ResultScalarType; 

public:

    void evaluateIdentity
    (const Plato::SpatialModel                         & aSpatialModel,
     const Plato::ScalarMultiVectorT<StateScalarType>  & aState,
           Plato::ScalarMultiVectorT<ResultScalarType> & aResult)
    {
        auto tNumCells = aSpatialModel.Mesh->NumElements();

        Kokkos::parallel_for("identity residual", Kokkos::RangePolicy<int>(0,tNumCells), 
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal)
        {
            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerCell; tNode++)
            {
                for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
                {
                    Plato::OrdinalType tLocalOrdinal = tNode * ElementType::mNumDofsPerNode + tDof;

                    aResult(iCellOrdinal, tLocalOrdinal) = aState(iCellOrdinal,tLocalOrdinal);
                }
            }

        });
    }

    void evaluateInterpolate
    (const Plato::SpatialModel                         & aSpatialModel,
     const Plato::ScalarMultiVectorT<StateScalarType>  & aState,
           Plato::ScalarMultiVectorT<ResultScalarType> & aResult)
    {
        auto tNumCells = aSpatialModel.Mesh->NumElements();

        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Plato::InterpolateFromNodal<ElementType, ElementType::mNumDofsPerNode, /*offset=*/0, ElementType::mNumDofsPerNode> interpolateFromNodal;

        Kokkos::parallel_for("interpolate residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tCubPoint = tCubPoints(iGPOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            Plato::Array<ElementType::mNumSpatialDims, StateScalarType> tDisp;
            interpolateFromNodal(iCellOrdinal, tBasisValues, aState, tDisp);

            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerCell; tNode++)
            {
                for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
                {
                    Plato::OrdinalType tLocalOrdinal = tNode * ElementType::mNumDofsPerNode + tDof;

                    ResultScalarType tResult = tDisp(tDof)/tNumPoints;
                    Kokkos::atomic_add(&aResult(iCellOrdinal, tLocalOrdinal), tResult);
                }
            }

        });
    }

    void evaluateNonlocal
    (const Plato::SpatialModel                         & aSpatialModel,
     const std::string                                 & aSideSet,
           Plato::OrdinalType                            aContributingCell,
     const Plato::ScalarMultiVectorT<StateScalarType>  & aState,
           Plato::ScalarMultiVectorT<ResultScalarType> & aResult)
    {
    // just evaluating displacement at cubature point of provided contributing cell ("parent element")
    // and interpolating over face side set face elements
        auto tElementOrds   = aSpatialModel.Mesh->GetSideSetElements(aSideSet);
        auto tLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(aSideSet);
        Plato::OrdinalType tNumFaces = tElementOrds.size();

        auto tCubaturePoints  = ElementType::getCubPoints();
        auto tCubatureWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubatureWeights.size();

        Plato::InterpolateFromNodal<ElementType, ElementType::mNumDofsPerNode, /*offset=*/0, ElementType::mNumDofsPerNode> interpolateFromNodal;

        Kokkos::parallel_for("contact force", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tCellOrdinal = tElementOrds(iCellOrdinal);
            auto tCubaturePoint = tCubaturePoints(iGPOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubaturePoint);

            Plato::Array<ElementType::mNumSpatialDims, StateScalarType> tDisp;
            interpolateFromNodal(aContributingCell, tBasisValues, aState, tDisp);

            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
            {
                auto tLocalNodeOrd = tLocalNodeOrds(iCellOrdinal*ElementType::mNumNodesPerFace+tNode);

                for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
                {
                    auto tElementDofOrdinal = tLocalNodeOrd * ElementType::mNumDofsPerNode + tDof;

                    ResultScalarType tResult = tDisp(tDof)/tNumPoints;
                    Kokkos::atomic_add(&aResult(tCellOrdinal, tElementDofOrdinal), tResult);
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

// #ifdef PLATO_MESHMAP
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

void
check_element_type_is_tet(Plato::Mesh aMesh)
{
    auto tElementType = aMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")
}

Plato::SpatialModel
setup_2box_spatial_model(Plato::Mesh aMesh)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(aMesh, *tInputs, tDataMap);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(aMesh);

    auto tPairs = Plato::Contact::parse_contact(tInputs->sublist("Contact"), aMesh);
    Plato::Contact::set_parent_data_for_pairs<ElementType>(tPairs, tSpatialModel);

    tSpatialModel.addContact(tPairs);

    return tSpatialModel;
}

// #endif

// testing this to have as reference for the actual assembly tests below
//
TEUCHOS_UNIT_TEST(BoxMeshWidth1Tests, Connectivity)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // check connectivity
    auto tConnectivity_Host = Plato::TestHelpers::get( tMesh->Connectivity() );
    std::vector<Plato::OrdinalType> tConnectivity_Gold = {
        0, 6, 2, 7,
        0, 2, 3, 7,
        0, 3, 1, 7,
        0, 1, 5, 7,
        0, 5, 4, 7,
        0, 4, 6, 7};
    for(int iVal=0; iVal<tConnectivity_Gold.size(); iVal++){
        TEST_EQUALITY(tConnectivity_Host(iVal), tConnectivity_Gold[iVal]);
    }
}

// testing these to have as reference for the actual assembly tests below
//
TEUCHOS_UNIT_TEST(BoxMeshWidth1Tests, BlockMatrixRowAndColumnMaps)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    // check row map
    auto tRowMap_Host = Plato::TestHelpers::get( tJacobianMat->rowMap() );
    std::vector<Plato::OrdinalType> tRowMap_Gold = {0, 8, 13, 18, 23, 28, 33, 38, 46};
    for(int iVal=0; iVal<tRowMap_Gold.size(); iVal++){
        TEST_EQUALITY(tRowMap_Host(iVal), tRowMap_Gold[iVal]);
    }

    // check column indices
    auto tColumnIndices_Host = Plato::TestHelpers::get( tJacobianMat->columnIndices() );
    std::vector<Plato::OrdinalType> tColumnIndices_Gold = {
        0, 1, 2, 3, 4, 5, 6, 7, 
        0, 1, 3, 5, 7,
        0, 2, 3, 6, 7,
        0, 1, 2, 3, 7,
        0, 4, 5, 6, 7,
        0, 1, 4, 5, 7,
        0, 2, 4, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7
        };
    for(int iVal=0; iVal<tColumnIndices_Gold.size(); iVal++){
        TEST_EQUALITY(tColumnIndices_Host(iVal), tColumnIndices_Gold[iVal]);
    }
}

TEUCHOS_UNIT_TEST(BlockMatrixEntryOrdinalTests, OrdinalsMatchExpected)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    Plato::BlockMatrixEntryOrdinal<tNumNodesPerCell, tNumDofsPerNode, tNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );
    
    // test entry ordinals for different inputs
    std::vector<Plato::OrdinalType> tCells = {0, 5, 2, 4};
    std::vector<Plato::OrdinalType> tLocalDofsI = {0, 3, 11, 7};
    std::vector<Plato::OrdinalType> tLocalDofsJ = {10, 8, 2, 5};

    auto dCells = Plato::TestHelpers::create_device_view(tCells);
    auto dLocalDofsI = Plato::TestHelpers::create_device_view(tLocalDofsI);
    auto dLocalDofsJ = Plato::TestHelpers::create_device_view(tLocalDofsJ);

    std::vector<Plato::OrdinalType> tOrdinals_Gold = {64, 236, 350, 230};
    Plato::OrdinalVector tEntryOrds("store entry ordinals", tCells.size());

    // PARALLEL FOR
    Kokkos::parallel_for("get entry ordinals", Kokkos::RangePolicy<Plato::OrdinalType>(0,tCells.size()), KOKKOS_LAMBDA(Plato::OrdinalType iOrd)
    {
        tEntryOrds(iOrd) = tJacobianMatEntryOrdinal(dCells(iOrd), dLocalDofsI(iOrd), dLocalDofsJ(iOrd));
    });

    auto tEntryOrds_Host = Plato::TestHelpers::get( tEntryOrds );
    for(int iOrd=0; iOrd<tOrdinals_Gold.size(); iOrd++)
        TEST_EQUALITY(tEntryOrds_Host(iOrd), tOrdinals_Gold[iOrd]);
}

TEUCHOS_UNIT_TEST(JacobianTests, ElementDerivativesAreIdentity)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;

    constexpr int tSpaceDim = ElementType::mNumSpatialDims;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr int tNumDofsPerCell  = ElementType::mNumDofsPerCell;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);
    auto tDomain = tSpatialModel.Domains.front(); // only one domain
    auto tNumCells = tDomain.numCells();

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);

    // create dummy displacement workset
    std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::ScalarMultiVectorT<EvaluationType::StateScalarType> tDispWS("state workset", tNumCells, tNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS, tDomain);

    // evaluate jacobian
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    Plato::ScalarMultiVectorT<EvaluationType::ResultScalarType> tJacobian("JacobianState", tNumCells, tNumDofsPerCell);

    DummyResidual<EvaluationType> tResidual;
    tResidual.evaluateIdentity(tSpatialModel,tDispWS,tJacobian);

    // assemble
    Plato::BlockMatrixEntryOrdinal<tNumNodesPerCell, tNumDofsPerNode, tNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

    auto tJacobianMatEntries = tJacobianMat->entries();
    tWorksetBase.assembleJacobianFad
        (tNumDofsPerCell, tNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);

    auto tJacobianEntries_Host = Plato::TestHelpers::get( tJacobianMatEntries );

    // test assembled jacobian
    std::vector<Plato::Scalar> tJacobianEntries_Gold = { 
        6, 0, 0, 0, 6, 0, 0, 0, 6,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 2, 0, 0, 0, 2,
        0, 0, 0, 0, 0, 0, 0, 0, 0,

        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        6, 0, 0, 0, 6, 0, 0, 0, 6
        };

    for(int iVal=0; iVal<tJacobianEntries_Gold.size(); iVal++){
        TEST_FLOATING_EQUALITY(tJacobianEntries_Host(iVal), tJacobianEntries_Gold[iVal], 1e-12);
    }

}

TEUCHOS_UNIT_TEST(JacobianTests, ElementDerivativesAreShapeFunctions)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;

    constexpr int tSpaceDim = ElementType::mNumSpatialDims;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr int tNumDofsPerCell  = ElementType::mNumDofsPerCell;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);
    auto tNumCells = tMesh->NumElements();

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);

    // create dummy displacement workset
    std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::ScalarMultiVectorT<EvaluationType::StateScalarType> tDispWS("state workset", tNumCells, tNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // evaluate jacobian
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    Plato::ScalarMultiVectorT<EvaluationType::ResultScalarType> tJacobian("JacobianState", tNumCells, tNumDofsPerCell);

    DummyResidual<EvaluationType> tResidual;
    tResidual.evaluateInterpolate(tSpatialModel,tDispWS,tJacobian);

    // assemble
    Plato::BlockMatrixEntryOrdinal<tNumNodesPerCell, tNumDofsPerNode, tNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

    auto tJacobianMatEntries = tJacobianMat->entries();
    tWorksetBase.assembleJacobianFad
        (tNumDofsPerCell, tNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

    auto tJacobianEntries_Host = Plato::TestHelpers::get( tJacobianMatEntries );

    // test assembled jacobian
    std::vector<Plato::Scalar> tJacobianEntries_Gold = { 
        1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5
        };

    for(int iVal=0; iVal<tJacobianEntries_Gold.size(); iVal++){
        TEST_FLOATING_EQUALITY(tJacobianEntries_Host(iVal), tJacobianEntries_Gold[iVal], 1e-12);
    }

}

// #ifdef PLATO_MESHMAP

// testing mesh for contact as reference for the actual assembly tests below
//
TEUCHOS_UNIT_TEST(TwoBoxMeshWidth1Tests, Connectivity)
{
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    // check connectivity
    auto tConnectivity_Host = Plato::TestHelpers::get( tMesh->Connectivity() );
    std::vector<Plato::OrdinalType> tConnectivity_Gold = {
        0, 1, 2, 3,
        0, 1, 3, 4,
        0, 5, 6, 3,
        0, 5, 3, 2,
        0, 7, 5, 2,
        0, 7, 2, 1,

        8, 9, 10, 11,
        8, 9, 11, 12,
        8, 13, 14, 11,
        8, 13, 11, 10,
        8, 15, 13, 10,
        8, 15, 10, 9};
    for(int iVal=0; iVal<tConnectivity_Gold.size(); iVal++){
        TEST_EQUALITY(tConnectivity_Host(iVal), tConnectivity_Gold[iVal]);
    }

}

// testing mesh for contact as reference for the actual assembly tests below
//
TEUCHOS_UNIT_TEST(TwoBoxMeshWidth1Tests, BlockMatrixRowAndColumnMaps)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    // check row map
    auto tRowMap_Host = Plato::TestHelpers::get( tJacobianMat->rowMap() );
    std::vector<Plato::OrdinalType> tRowMap_Gold = {
        0, 8, 14, 20, 27, 31, 37, 41, 46,
        54, 60, 66, 73, 77, 83, 87, 92};
    for(int iVal=0; iVal<tRowMap_Gold.size(); iVal++){
        TEST_EQUALITY(tRowMap_Host(iVal), tRowMap_Gold[iVal]);
    }

    // check column indices
    auto tColumnIndices_Host = Plato::TestHelpers::get( tJacobianMat->columnIndices() );
    std::vector<Plato::OrdinalType> tColumnIndices_Gold = {
        0, 1, 2, 3, 4, 5, 6, 7, 
        0, 1, 2, 3, 4, 7,
        0, 1, 2, 3, 5, 7,
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 3, 4,
        0, 2, 3, 5, 6, 7,
        0, 3, 5, 6,
        0, 1, 2, 5, 7,

        8, 9, 10, 11, 12, 13, 14, 15,
        8, 9, 10, 11, 12, 15,
        8, 9, 10, 11, 13, 15,
        8, 9, 10, 11, 12, 13, 14,
        8, 9, 11, 12,
        8, 10, 11, 13, 14, 15,
        8, 11, 13, 14,
        8, 9, 10, 13, 15
        };
    for(int iVal=0; iVal<tColumnIndices_Gold.size(); iVal++){
        TEST_EQUALITY(tColumnIndices_Host(iVal), tColumnIndices_Gold[iVal]);
    }
        
}

TEUCHOS_UNIT_TEST(ContactNodeNodeMapTests, AddContactContributionsToNodeMap)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;

    auto tSpatialModel = setup_2box_spatial_model(tMesh);

    // add contact graph through spatial model when constructing block matrix
    Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );
    
    auto tFullOffsetMap = tJacobian->rowMap();
    auto tFullNodeOrds  = tJacobian->columnIndices();

    // check row map
    auto tRowMap_Host = Plato::TestHelpers::get( tFullOffsetMap );
    std::vector<Plato::OrdinalType> tRowMap_Gold = {
        0, 13, 19, 25, 32, 36, 47, 56, 66,
        74, 87, 99, 113, 124, 130, 134, 139};

    for(int iVal=0; iVal<tRowMap_Gold.size(); iVal++){
        TEST_EQUALITY(tRowMap_Host(iVal), tRowMap_Gold[iVal]);
    }

    // check column indices
    auto tColumnIndices_Host = Plato::TestHelpers::get( tFullNodeOrds );
    std::vector<Plato::OrdinalType> tColumnIndices_Gold = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        0, 1, 2, 3, 4, 7,
        0, 1, 2, 3, 5, 7,
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 3, 4,
        0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12,
        0, 3, 5, 6, 8, 9, 10, 11, 12,
        0, 1, 2, 5, 7, 8, 9, 10, 11, 12,

        8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15,
        0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 15,
        0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12,
        8, 10, 11, 13, 14, 15,
        8, 11, 13, 14,
        8, 9, 10, 13, 15
        };

    for(int iVal=0; iVal<tColumnIndices_Gold.size(); iVal++){
        TEST_EQUALITY(tColumnIndices_Host(iVal), tColumnIndices_Gold[iVal]);
    }
}

TEUCHOS_UNIT_TEST(ContactNodeNodeMapTests, TransposeNodeMapWithContactContributions)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = get_2box_mesh_params();

    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;

    auto tSpatialModel = setup_2box_spatial_model(tMesh);

    // add contact graph through spatial model when constructing block matrix transpose
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianT =
        Plato::CreateBlockMatrixTranspose<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    auto tFullOffsetMap = tJacobianT->rowMap();
    auto tFullNodeOrds  = tJacobianT->columnIndices();

    // check row map
    auto tRowMap_Host = Plato::TestHelpers::get( tFullOffsetMap );
    std::vector<Plato::OrdinalType> tRowMap_Gold = {
        0, 12, 21, 31, 42, 46, 56, 64, 73, 
        85, 95, 105, 116, 124, 130, 134, 139};

    for(int iVal=0; iVal<tRowMap_Gold.size(); iVal++){
        TEST_EQUALITY(tRowMap_Host(iVal), tRowMap_Gold[iVal]);
    }

    // check column indices
    auto tColumnIndices_Host = Plato::TestHelpers::get( tFullNodeOrds );
    std::vector<Plato::OrdinalType> tColumnIndices_Gold = {
        0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12,
        0, 1, 2, 3, 4, 7, 9, 11, 12,
        0, 1, 2, 3, 5, 7, 9, 10, 11, 12,
        0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12,
        0, 1, 3, 4,
        0, 2, 3, 5, 6, 7, 9, 10, 11, 12,
        0, 3, 5, 6, 9, 10, 11, 12,
        0, 1, 2, 5, 7, 9, 10, 11, 12,

        0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 5, 6, 7, 8, 9, 10, 11, 12, 15,
        0, 5, 6, 7, 8, 9, 10, 11, 13, 15,
        0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
        0, 5, 6, 7, 8, 9, 11, 12,
        8, 10, 11, 13, 14, 15,
        8, 11, 13, 14,
        8, 9, 10, 13, 15
        };

    for(int iVal=0; iVal<tColumnIndices_Gold.size(); iVal++){
        TEST_EQUALITY(tColumnIndices_Host(iVal), tColumnIndices_Gold[iVal]);
    }
}

TEUCHOS_UNIT_TEST(BlockMatrixEntryOrdinalTests, OrdinalsMatchExpected_LocalOrdinalsWithContact)
{
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;

    Plato::SpatialModel tSpatialModel = setup_2box_spatial_model(tMesh);

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    Plato::BlockMatrixEntryOrdinal<tNumNodesPerCell, tNumDofsPerNode, tNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );
    
    // test entry ordinals for different inputs
    std::vector<Plato::OrdinalType> tCells = {8, 5, 10, 4};
    std::vector<Plato::OrdinalType> tLocalDofsI = {0, 3, 11, 7};
    std::vector<Plato::OrdinalType> tLocalDofsJ = {10, 8, 2, 5};

    auto dCells = Plato::TestHelpers::create_device_view(tCells);
    auto dLocalDofsI = Plato::TestHelpers::create_device_view(tLocalDofsI);
    auto dLocalDofsJ = Plato::TestHelpers::create_device_view(tLocalDofsJ);

    std::vector<Plato::OrdinalType> tOrdinals_Gold = {622, 524, 845, 374};
    Plato::OrdinalVector tEntryOrds("store entry ordinals", tCells.size());

    // PARALLEL FOR
    Kokkos::parallel_for("get entry ordinals", Kokkos::RangePolicy<Plato::OrdinalType>(0,tCells.size()), KOKKOS_LAMBDA(Plato::OrdinalType iOrd)
    {
        tEntryOrds(iOrd) = tJacobianMatEntryOrdinal(dCells(iOrd), dLocalDofsI(iOrd), dLocalDofsJ(iOrd));
    });

    auto tEntryOrds_Host = Plato::TestHelpers::get( tEntryOrds );
    for(int iOrd=0; iOrd<tOrdinals_Gold.size(); iOrd++)
        TEST_EQUALITY(tEntryOrds_Host(iOrd), tOrdinals_Gold[iOrd]);
}

TEUCHOS_UNIT_TEST(BlockMatrixEntryOrdinalTests, OrdinalsMatchExpected_NonLocalOrdinalsWithContact)
{
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr int tNumNodesPerFace  = ElementType::mNumNodesPerFace;

    Plato::SpatialModel tSpatialModel = setup_2box_spatial_model(tMesh);

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    Plato::BlockMatrixEntryOrdinal<tNumNodesPerCell, tNumDofsPerNode, tNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );
    
    // test entry ordinals for different inputs
    auto tPairs = tSpatialModel.contactPairs();
    auto tPair = tPairs[0];
    auto tChildElements = tPair.surfaceA.childElements();
    auto tChildFaceLocalNodes = tPair.surfaceA.childFaceLocalNodes();
    auto tParentElements = tPair.surfaceA.parentElements();
    auto tElementWiseChildMap = tPair.surfaceA.elementWiseChildMap();

    std::vector<Plato::OrdinalType> tLocalNodeOrdsI = {0, 2};
    std::vector<Plato::OrdinalType> tNodeDofsI = {1, 0};
    std::vector<Plato::OrdinalType> tLocalDofsJ = {10, 5};

    auto dLocalNodeOrdsI = Plato::TestHelpers::create_device_view(tLocalNodeOrdsI);
    auto dNodeDofsI = Plato::TestHelpers::create_device_view(tNodeDofsI);
    auto dLocalDofsJ = Plato::TestHelpers::create_device_view(tLocalDofsJ);

    Plato::OrdinalVector tEntryOrds("store entry ordinals", tChildElements.size());

    // PARALLEL FOR
    Kokkos::parallel_for("get entry ordinals", Kokkos::RangePolicy<Plato::OrdinalType>(0,tChildElements.size()), KOKKOS_LAMBDA(Plato::OrdinalType iOrd)
    {
        auto tChildElement = tChildElements(iOrd);
        auto tLocalNodeOrd = tChildFaceLocalNodes(dLocalNodeOrdsI(iOrd));
        auto tParentOrd = tElementWiseChildMap(iOrd * tNumNodesPerFace + dLocalNodeOrdsI(iOrd));
        auto tParentElement = tParentElements(tParentOrd);
        auto tLocalDofI = tLocalNodeOrd * tNumDofsPerNode + dNodeDofsI(iOrd);
        tEntryOrds(iOrd) = tJacobianMatEntryOrdinal(tChildElement, tParentElement, tLocalDofI, dLocalDofsJ(iOrd));
    });

    // Notes to decipher how gold values were computed:
    //  * Child elements: 2, 4
    //  * Child face local nodes: 0, 2, 1
    //  * Child nodes:                   0, 5, 6, 7
    //  * Corresponding Parent elements: 7, 6, 6, 6
    std::vector<Plato::OrdinalType> tOrdinals_Gold = {112, 560};

    auto tEntryOrds_Host = Plato::TestHelpers::get( tEntryOrds );
    for(int iOrd=0; iOrd<tOrdinals_Gold.size(); iOrd++)
        TEST_EQUALITY(tEntryOrds_Host(iOrd), tOrdinals_Gold[iOrd]);
}

TEUCHOS_UNIT_TEST(JacobianTestsWithContactGraph, ElementDerivativesAreShapeFunctions)
{
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;

    constexpr int tSpaceDim = ElementType::mNumSpatialDims;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr int tNumDofsPerCell  = ElementType::mNumDofsPerCell;

    Plato::SpatialModel tSpatialModel = setup_2box_spatial_model(tMesh);
    auto tNumCells = tMesh->NumElements();

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);

    // create dummy displacement workset
    std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::ScalarMultiVectorT<EvaluationType::StateScalarType> tDispWS("state workset", tNumCells, tNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // evaluate jacobian
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    Plato::ScalarMultiVectorT<EvaluationType::ResultScalarType> tJacobian("JacobianState", tNumCells, tNumDofsPerCell);

    DummyResidual<EvaluationType> tResidual;
    tResidual.evaluateInterpolate(tSpatialModel,tDispWS,tJacobian);

    // assemble
    Plato::BlockMatrixEntryOrdinal<tNumNodesPerCell, tNumDofsPerNode, tNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

    auto tJacobianMatEntries = tJacobianMat->entries();
    tWorksetBase.assembleJacobianFad
        (tNumDofsPerCell, tNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

    auto tJacobianEntries_Host = Plato::TestHelpers::get( tJacobianMatEntries );

    // test assembled jacobian
    std::vector<Plato::Scalar> tJacobianEntries_Gold = { 
        1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,

        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
 
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,

        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,

        



        1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        };

    for(int iVal=0; iVal<tJacobianEntries_Gold.size(); iVal++){
        TEST_FLOATING_EQUALITY(tJacobianEntries_Host(iVal), tJacobianEntries_Gold[iVal], 1e-12);
    }

}

TEUCHOS_UNIT_TEST(JacobianTestsWithContactGraph, ElementDerivativesAreShapeFunctions_NonlocalContributionsAreShapeFunctions)
{
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    check_element_type_is_tet(tMesh);
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;

    constexpr int tSpaceDim = ElementType::mNumSpatialDims;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr int tNumDofsPerCell  = ElementType::mNumDofsPerCell;

    Plato::SpatialModel tSpatialModel = setup_2box_spatial_model(tMesh);
    auto tNumCells = tMesh->NumElements();

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);

    // create dummy displacement workset
    std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    auto u = Plato::TestHelpers::create_device_view(u_host);

    Plato::ScalarMultiVectorT<EvaluationType::StateScalarType> tDispWS("state workset", tNumCells, tNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // evaluate jacobian for volumetric terms
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tSpatialModel );

    Plato::ScalarMultiVectorT<EvaluationType::ResultScalarType> tJacobian("JacobianState", tNumCells, tNumDofsPerCell);

    DummyResidual<EvaluationType> tResidual;
    tResidual.evaluateInterpolate(tSpatialModel,tDispWS,tJacobian);

    // assemble volumetric terms
    Plato::BlockMatrixEntryOrdinal<tNumNodesPerCell, tNumDofsPerNode, tNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

    auto tJacobianMatEntries = tJacobianMat->entries();
    tWorksetBase.assembleJacobianFad
        (tNumDofsPerCell, tNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

    // get pair data
    auto tPairs = tSpatialModel.contactPairs();
    auto tPair = tPairs[0];

    // evaluate and assemble (nonlocal) surface terms for node on surface A
    Plato::OrdinalType tContributingCell = 7; // "parent" element in which to compute displacement
    Plato::OrdinalType tContributingNode = 0; // local child node corresponding to contributing parent element
    auto tSideSet = tPair.surfaceA.childSideSet();
    auto tChildCells = tPair.surfaceA.childElements();
    auto tParentCells = tPair.surfaceA.parentElements();
    auto tElementWiseChildMap = tPair.surfaceA.elementWiseChildMap();
    auto tChildFaceLocalNodes = tPair.surfaceA.childFaceLocalNodes();

    Plato::ScalarMultiVectorT<EvaluationType::ResultScalarType> tNonlocalJacobianA("JacobianState", tNumCells, tNumDofsPerCell);
    tResidual.evaluateNonlocal(tSpatialModel,tSideSet,tContributingCell,tDispWS,tNonlocalJacobianA);

    tWorksetBase.assembleJacobianFad
        (tNumDofsPerCell, tChildCells, tParentCells, tElementWiseChildMap, tChildFaceLocalNodes, tContributingNode, tJacobianMatEntryOrdinal, tNonlocalJacobianA, tJacobianMatEntries);

    // evaluate and assemble (nonlocal) surface terms for node on surface B
    tContributingCell = 4; // "parent" element in which to compute displacement
    tContributingNode = 0; // local child node corresponding to contributing parent element
    tSideSet = tPair.surfaceB.childSideSet();
    tChildCells = tPair.surfaceB.childElements();
    tParentCells = tPair.surfaceB.parentElements();
    tElementWiseChildMap = tPair.surfaceB.elementWiseChildMap();
    tChildFaceLocalNodes = tPair.surfaceB.childFaceLocalNodes();

    Plato::ScalarMultiVectorT<EvaluationType::ResultScalarType> tNonlocalJacobianB("JacobianState", tNumCells, tNumDofsPerCell);
    tResidual.evaluateNonlocal(tSpatialModel,tSideSet,tContributingCell,tDispWS,tNonlocalJacobianB);

    tWorksetBase.assembleJacobianFad
        (tNumDofsPerCell, tChildCells, tParentCells, tElementWiseChildMap, tChildFaceLocalNodes, tContributingNode, tJacobianMatEntryOrdinal, tNonlocalJacobianB, tJacobianMatEntries);

    // test assembled jacobian
    auto tJacobianEntries_Host = Plato::TestHelpers::get( tJacobianMatEntries );
    std::vector<Plato::Scalar> tJacobianEntries_Gold = { 
        1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
 
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        



        1.5, 0, 0, 0, 1.5, 0, 0, 0, 1.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.75, 0, 0, 0, 0.75, 0, 0, 0, 0.75,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,

        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        0.25, 0, 0, 0, 0.25, 0, 0, 0, 0.25,
        0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5,
        };

    for(int iVal=0; iVal<tJacobianEntries_Gold.size(); iVal++){
        TEST_FLOATING_EQUALITY(tJacobianEntries_Host(iVal), tJacobianEntries_Gold[iVal], 1e-12);
    }

}

// #endif
}
