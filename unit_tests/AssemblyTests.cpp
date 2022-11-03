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

#include "InterpolateFromNodal.hpp"

#ifdef PLATO_MESHMAP
#include "ContactUtils.hpp"
#include "UpdateGraphForContact.hpp"
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

    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tMesh );

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
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tCells.size()), KOKKOS_LAMBDA(Plato::OrdinalType iOrd)
    {
        tEntryOrds(iOrd) = tJacobianMatEntryOrdinal(dCells(iOrd), dLocalDofsI(iOrd), dLocalDofsJ(iOrd));
    }, "get entry ordinals");

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
    Plato::BlockMatrixEntryOrdinal<tNumNodesPerCell, tNumDofsPerNode, tNumDofsPerNode>
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

#ifdef PLATO_MESHMAP
TEUCHOS_UNIT_TEST(ContactNodeNodeMapTests, AddContactContributionsToNodeMap)
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
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs, tDataMap);

    // parse contact
    auto tPairs = Plato::Contact::parse_contact(*tInputs, tMesh);

    // get full arrays of child nodes and parent elements
    auto tNumTotalNodes = Plato::Contact::count_total_child_nodes(tPairs);

    Plato::OrdinalVector tAllChildNodes("", tNumTotalNodes);
    Plato::OrdinalVector tAllParentElements("", tNumTotalNodes);
    Plato::Contact::populate_full_contact_arrays<ElementType>(tPairs, tSpatialModel, tAllChildNodes, tAllParentElements);
    Plato::Contact::check_for_repeated_child_nodes(tAllChildNodes,tMesh);

    // add contact graph
    Plato::Contact::UpdateGraphForContact updateGraphForContact(tMesh, tAllChildNodes, tAllParentElements);

    Teuchos::RCP<Plato::CrsMatrixType> tJacobianOrig =
        Plato::CreateBlockMatrix<Plato::CrsMatrixType, tNumDofsPerNode, tNumDofsPerNode>( tMesh );
    
    auto tJacobian = updateGraphForContact(tJacobianOrig);
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
#endif

}
