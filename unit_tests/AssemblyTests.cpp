#include "util/PlatoTestHelpers.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "PlatoUtilities.hpp"

#include "PlatoStaticsTypes.hpp"
#include "Tet4.hpp"
#include "MechanicsElement.hpp"

#include "EngineMesh.hpp"
#include "SpatialModel.hpp"
#include "Plato_MeshMap.hpp"

#include "elliptic/EvaluationTypes.hpp"

#include "WorksetBase.hpp"
#include "ImplicitFunctors.hpp"

#include "InterpolateFromNodal.hpp"

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

        auto tCubPoints  = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tCubPoint = tCubPoints(iGPOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerCell; tNode++)
            {
                for( Plato::OrdinalType tDof=0; tDof<ElementType::mNumDofsPerNode; tDof++)
                {
                    Plato::OrdinalType tLocalOrdinal = tNode * ElementType::mNumDofsPerNode + tDof;

                    ResultScalarType tResult = aState(iCellOrdinal,tLocalOrdinal);
                    Kokkos::atomic_add(&aResult(iCellOrdinal, tLocalOrdinal), tResult);
                }
            }

        }, "identity residual");
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

        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, tNumPoints}),
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

                    ResultScalarType tResult = tDisp(tDof);
                    Kokkos::atomic_add(&aResult(iCellOrdinal, tLocalOrdinal), tResult);
                }
            }

        }, "interpolate residual");
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

Plato::SpatialModel
setup_2box_spatial_model(Plato::Mesh aMesh)
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

// testing this to have as reference for the actual assembly tests below
//
TEUCHOS_UNIT_TEST(BoxMeshWidth1Tests, Connectivity)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

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

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tMesh );

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

// testing mesh for contact as reference for the actual assembly tests below
//
TEUCHOS_UNIT_TEST(TwoBoxMeshWidth1Tests, Connectivity)
{
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tElementType = tMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")

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
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tElementType = tMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tMesh );

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

TEUCHOS_UNIT_TEST(BlockMatrixEntryOrdinalTests, OrdinalsMatchExpected)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    constexpr int tSpaceDim = ElementType::mNumSpatialDims;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tMesh );

    Plato::BlockMatrixEntryOrdinal<tSpaceDim, tNumDofsPerNode, tNumDofsPerNode, tNumNodesPerCell>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );
    
    // test entry ordinals for different inputs
    std::vector<Plato::OrdinalType> tCells = {0, 5, 2, 4};
    std::vector<Plato::OrdinalType> tLocalDofsI = {0, 3, 11, 7};
    std::vector<Plato::OrdinalType> tLocalDofsJ = {10, 8, 2, 5};

    std::vector<Plato::OrdinalType> tOrdinals_Gold = {64, 236, 350, 230};

    for(int iOrd=0; iOrd<tOrdinals_Gold.size(); iOrd++)
    {
        Plato::OrdinalType tEntryOrdinal = tJacobianMatEntryOrdinal(tCells[iOrd], tLocalDofsI[iOrd], tLocalDofsJ[iOrd]);
        TEST_EQUALITY(tEntryOrdinal, tOrdinals_Gold[iOrd]);
    }
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
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::ScalarMultiVectorT<EvaluationType::StateScalarType> tDispWS("state workset", tNumCells, tNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS, tDomain);

    // evaluate jacobian
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tMesh );

    Plato::ScalarMultiVectorT<EvaluationType::ResultScalarType> tJacobian("JacobianState", tNumCells, tNumDofsPerCell);

    DummyResidual<EvaluationType> tResidual;
    tResidual.evaluateIdentity(tSpatialModel,tDispWS,tJacobian);

    // assemble
    Plato::BlockMatrixEntryOrdinal<tSpaceDim, tNumDofsPerNode, tNumDofsPerNode, tNumNodesPerCell>
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
    auto tDomain = tSpatialModel.Domains.front(); // only one domain
    auto tNumCells = tDomain.numCells();

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);

    // create dummy displacement workset
    std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::ScalarMultiVectorT<EvaluationType::StateScalarType> tDispWS("state workset", tNumCells, tNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS, tDomain);

    // evaluate jacobian
    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tMesh );

    Plato::ScalarMultiVectorT<EvaluationType::ResultScalarType> tJacobian("JacobianState", tNumCells, tNumDofsPerCell);

    DummyResidual<EvaluationType> tResidual;
    tResidual.evaluateInterpolate(tSpatialModel,tDispWS,tJacobian);

    // assemble
    Plato::BlockMatrixEntryOrdinal<tSpaceDim, tNumDofsPerNode, tNumDofsPerNode, tNumNodesPerCell>
        tJacobianMatEntryOrdinal( tJacobianMat, tMesh );

    auto tJacobianMatEntries = tJacobianMat->entries();
    tWorksetBase.assembleJacobianFad
        (tNumDofsPerCell, tNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);

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

TEUCHOS_UNIT_TEST(ContactNodeNodeMapTests, WhateverINeedItToBeForNow)
{
    std::string tMeshName = "two_block_contact.exo";
    auto tMesh = std::make_shared<Plato::EngineMesh>(tMeshName);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    auto tElementType = tMesh->ElementType();
    if( Plato::tolower(tElementType) != "tetra"  &&
        Plato::tolower(tElementType) != "tetra4" &&
        Plato::tolower(tElementType) != "tet4" )
        ANALYZE_THROWERR("AssemblyTests: Mesh element type being used is not tet4")
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tSpaceDim = ElementType::mNumSpatialDims;

    Plato::SpatialModel tSpatialModel = setup_2box_spatial_model(tMesh);

    // get side set info
    std::string tSideSetName = "block1_child";
    auto tChildFaceNodes = tMesh->GetNodeSetNodes(tSideSetName);

    // check block 1 child face nodes
    auto tNumChildNodes = tChildFaceNodes.extent(0);
    TEST_EQUALITY(tNumChildNodes, 4);

    auto tChildFaceNodes_Host = Plato::TestHelpers::get( tChildFaceNodes );
    std::vector<Plato::OrdinalType> tChildFaceNodes_Gold = {0, 5, 6, 7};
    for(int iVal=0; iVal<tChildFaceNodes_Gold.size(); iVal++){
        TEST_EQUALITY(tChildFaceNodes_Host(iVal), tChildFaceNodes_Gold[iVal]);
    }

    // get child node locations and map them
    Plato::ScalarMultiVector tChildNodeCoords("child node locations", tSpaceDim, tNumChildNodes);
    Plato::ScalarMultiVector tChildNodeMappedCoords("mapped child node locations", tSpaceDim, tNumChildNodes);
    Plato::Array<tSpaceDim> tTranslation = {1.0, 0.0, 0.0};

    auto tCoords = tMesh->Coordinates();
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
    {
        auto tNodeOrdinal = tChildFaceNodes(nodeOrdinal);
        for (Plato::OrdinalType iDim = 0; iDim < tSpaceDim; iDim++)
        {
            tChildNodeCoords(iDim, nodeOrdinal) = tCoords(tNodeOrdinal*tSpaceDim+iDim);
            tChildNodeMappedCoords(iDim, nodeOrdinal) = tCoords(tNodeOrdinal*tSpaceDim+iDim) + tTranslation(iDim);
        }
    }, "get coords");

    // find parent elements
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // can pull this out of PBC MPCs? (don't want to repeat this code)
    // in fact, can just make a method in SpatialModel class to return cell map or domain class
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    std::string tParentDomainName = "block_2";
    Plato::OrdinalVector tDomainCellMap;
    bool tFindName = 0;
    for(auto& tDomain : tSpatialModel.Domains)
    {
        auto tName = tDomain.getElementBlockName();
        if( tName == tParentDomainName )
            tDomainCellMap = tDomain.cellOrdinals();
            tFindName = 1;
    }
    if( tFindName == 0 )
    {
        ANALYZE_THROWERR("Assembly Tests: PARENT DOMAIN FOR PBC MULTIPOINT CONSTRAINT NOT FOUND.")
    }

    Plato::OrdinalVector tParentElements("parent elements", tNumChildNodes);

    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
      (tSpatialModel.Mesh, tDomainCellMap, tChildNodeCoords, tChildNodeMappedCoords, tParentElements);

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // THIS NEEDS TO BE DONE AFTER DIAGONALS ARE INSERTED
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // Plato::OrdinalVectorT<const Plato::OrdinalType> tOffsetMap;
    // Plato::OrdinalVectorT<const Plato::OrdinalType> tNodeOrds;
    // tMesh->NodeNodeGraph(tOffsetMap, tNodeOrds);

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tMesh );
    auto tOffsetMap = tJacobianMat->rowMap();
    auto tNodeOrds  = tJacobianMat->columnIndices();

    // find and store number of entries in node node graph for just child nodes
    Plato::OrdinalVector tChildOffsetMap("offset map for just child nodes", tNumChildNodes+1);
    Plato::OrdinalType tNumChildConnectedNodes(0);

    Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumChildNodes),
    KOKKOS_LAMBDA (const Plato::OrdinalType& aOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        auto tChildNode = tChildFaceNodes(aOrdinal);
        const auto tNumConnected = tOffsetMap(tChildNode+1) - tOffsetMap(tChildNode);

        aUpdate += tNumConnected;
        if( tIsFinal )
        {
          tChildOffsetMap(aOrdinal+1) = aUpdate;
        }
    }, tNumChildConnectedNodes);
    
    // figure out number of nodes and save them to fill out maps
    auto tNumNodes = tMesh->NumNodes();
    Plato::OrdinalVector tMarkedNodes("marking child nodes", tNumNodes);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), tMarkedNodes);  

    auto tConnectivity = tMesh->Connectivity();

    Plato::OrdinalType tNumOrdinals = tNumChildConnectedNodes*ElementType::mNumNodesPerCell;
    Plato::OrdinalVector tFatGraph_ordinals("largest number of possible nodes in graph", tNumOrdinals);
    Plato::OrdinalVector tNumConnectedNodes("number of nodes connected by contact", tNumChildNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType iChildNode)
    {
        Plato::OrdinalType tNumUnique(0);

        auto tChildNode = tChildFaceNodes(iChildNode);
        tMarkedNodes(tChildNode) = iChildNode;
                
        Plato::OrdinalType tFrom = tOffsetMap(tChildNode);
        Plato::OrdinalType tTo   = tOffsetMap(tChildNode + 1);

        auto tFatGraphOffset = tChildOffsetMap(iChildNode)*ElementType::mNumNodesPerCell;

        for(Plato::OrdinalType iOrd=tFrom; iOrd<tTo; iOrd++)
        {
            auto tGraphNode = tNodeOrds(iOrd);
            
            // check if node in graph is a child node
            Plato::OrdinalType tOutput = -1;
            for(Plato::OrdinalType iChild=0; iChild<tNumChildNodes; iChild++)
            {
                if (tChildFaceNodes(iChild) == tGraphNode)
                {
                    tOutput = iChild;
                    break;
                }
            }

            if (tOutput >= 0)
            {
                auto tParentElement = tParentElements(tOutput);
                for(Plato::OrdinalType tElemLocalNodeOrd=0; tElemLocalNodeOrd<ElementType::mNumNodesPerCell; tElemLocalNodeOrd++)
                {
                    auto tNodeOrd = tConnectivity(tParentElement*ElementType::mNumNodesPerCell + tElemLocalNodeOrd);

                    // get unique parent nodes
                    bool isUnique = true;
                    for( Plato::OrdinalType tIndex=0; tIndex<tNumUnique; tIndex++ )
                    {
                        if( tFatGraph_ordinals(tFatGraphOffset+tIndex) == tNodeOrd )
                        {
                            isUnique = false;
                        }
                    }
                    if(isUnique)
                    {
                        tFatGraph_ordinals(tFatGraphOffset+tNumUnique) = tNodeOrd;
                        tNumUnique++;
                    }
                }
            }
        }
        tNumConnectedNodes(iChildNode) = tNumUnique;
    });

    Plato::OrdinalVector tFullOffsetMap("offset map accounting for contact", tNumNodes+1);

    Plato::OrdinalType tNumNodeNodeEntries(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumNodes),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        auto tChildMark = tMarkedNodes(iOrdinal);
        
        auto tOriginalNum = tOffsetMap(iOrdinal+1) - tOffsetMap(iOrdinal);
        auto tContactNum = tNumConnectedNodes(tChildMark);

        const auto tVal = (tChildMark < 0) ? tOriginalNum : tOriginalNum + tContactNum;
        aUpdate += tVal;
        if( tIsFinal )
        {
          tFullOffsetMap(iOrdinal+1) = aUpdate;
        }
    }, tNumNodeNodeEntries);

    Plato::OrdinalVector tFullNodeOrds("node-node ordinals accounting for contact", tNumNodeNodeEntries);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
    {
        auto tNewFrom = tFullOffsetMap(aNodeOrdinal);
        auto tNewNum  = tFullOffsetMap(aNodeOrdinal+1) - tNewFrom;

        // fill in old entries
        auto tOldFrom = tOffsetMap(aNodeOrdinal);
        auto tOldNum  = tOffsetMap(aNodeOrdinal+1) - tOldFrom;
        for( Plato::OrdinalType tIndex=0; tIndex<tOldNum; tIndex++ )
        {
            tFullNodeOrds(tNewFrom+tIndex) = tNodeOrds(tOldFrom+tIndex);
        }

        // fill in new entries
        auto tChildMark = tMarkedNodes(aNodeOrdinal);
        if (tChildMark >= 0)
        {
            auto tStart = tNewFrom + tOldNum;
            auto tEnd   = tStart + tNewNum;

            auto tFatGraphOffset = tChildOffsetMap(tChildMark)*ElementType::mNumNodesPerCell;
            for( Plato::OrdinalType tIndex=tStart; tIndex<tEnd; tIndex++ )
            {
                tFullNodeOrds(tIndex) = tFatGraph_ordinals(tFatGraphOffset++);
            }
        }
    }, "node ordinals accounting for contact");

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // then i need to sort? can pull out the sort from engine mesh?
    // PULL IT OUT INTO A UTILITY?
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
    // sort list of connected nodes (otherwise cpu and gpu builds produce different graphs)
    auto& tOffs = tFullOffsetMap;
    auto& tOrds = tFullNodeOrds;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
    {
        auto tFrom = tOffs(aNodeOrdinal);
        auto tTo = tOffs(aNodeOrdinal+1)-1;
        for( decltype(tFrom) tIndexI=tFrom; tIndexI<tTo; tIndexI++ )
        {
            for( decltype(tFrom) tIndexJ=tFrom; tIndexJ<tTo; tIndexJ++ )
            {
                if( tOrds(tIndexJ) > tOrds(tIndexJ+1) )
                {
                    auto tHereHoldThis = tOrds(tIndexJ+1);
                    tOrds(tIndexJ+1) = tOrds(tIndexJ);
                    tOrds(tIndexJ) = tHereHoldThis;
                }
            }
        }
    }, "sort ordinals");

    // check row map
    auto tRowMap_Host = Plato::TestHelpers::get( tFullOffsetMap );
    std::vector<Plato::OrdinalType> tRowMap_Gold = {
        0, 13, 19, 25, 32, 36, 47, 56, 66,
        74, 80, 86, 93, 97, 103, 107, 112};
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

}
