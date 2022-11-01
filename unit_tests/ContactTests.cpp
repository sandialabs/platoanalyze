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

namespace ContactTests
{

template <typename EvaluationType>
class AbstractSurfaceDisplacement : public EvaluationType::ElementType
{
protected:
    using ElementType = typename EvaluationType::ElementType;
    using InStateT  = typename EvaluationType::StateScalarType;  
    using OutStateT = typename EvaluationType::ResultScalarType; 

public:
    AbstractSurfaceDisplacement(Plato::Scalar aScale = 1.0) : mScale(aScale) 
    {}

    virtual ~AbstractSurfaceDisplacement(){}

    virtual KOKKOS_INLINE_FUNCTION void
    operator()
    (Plato::OrdinalType                                            aCellOrdinal, 
     const Plato::Array<ElementType::mNumNodesPerFace>           & aBasisFunctions,
     const Plato::ScalarMultiVectorT<InStateT>                   & aState,
           Plato::Array<ElementType::mNumSpatialDims, OutStateT> & aSurfaceDisp) const = 0;

protected:
    Plato::Scalar mScale;
};

template <typename EvaluationType,
          Plato::OrdinalType NumDofsPerNode = EvaluationType::ElementType::mNumSpatialDims>
class SurfaceDisplacement : 
    public AbstractSurfaceDisplacement<EvaluationType>
 {

private: 
    using ElementType = typename EvaluationType::ElementType;
    using InStateT    = typename EvaluationType::StateScalarType;  
    using OutStateT   = typename EvaluationType::ResultScalarType; 

    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerFace;

public:
    SurfaceDisplacement
     (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aSideSetElements,
      const Plato::OrdinalVectorT<const Plato::OrdinalType> & aSideSetLocalNodes,
      Plato::Scalar                                           aScale = 1.0) :
     AbstractSurfaceDisplacement<EvaluationType>(aScale),
     mSideSetElements(aSideSetElements),
     mSideSetLocalNodes(aSideSetLocalNodes)
    {
    }

    KOKKOS_INLINE_FUNCTION void
    operator()
    (Plato::OrdinalType                               aCellOrdinal, 
     const Plato::Array<mNumNodesPerFace>           & aBasisFunctions,
     const Plato::ScalarMultiVectorT<InStateT>      & aState,
           Plato::Array<mNumSpatialDims, OutStateT> & aSurfaceDisp) const override
    {
        auto tGlobalCellOrdinal = mSideSetElements(aCellOrdinal);

        auto tScale = this->mScale;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
        {
            aSurfaceDisp(tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerFace; tNodeIndex++)
            {
                Plato::OrdinalType tSurfaceNode = mSideSetLocalNodes(aCellOrdinal*mNumNodesPerFace + tNodeIndex);
                Plato::OrdinalType tCellDofIndex = NumDofsPerNode * tSurfaceNode + tDofIndex; 
                aSurfaceDisp(tDofIndex) += tScale * aBasisFunctions(tNodeIndex) * aState(tGlobalCellOrdinal, tCellDofIndex);
            }
        }
    }

private:
    Plato::OrdinalVectorT<const Plato::OrdinalType> mSideSetElements;
    Plato::OrdinalVectorT<const Plato::OrdinalType> mSideSetLocalNodes;

};

template <typename EvaluationType,
          Plato::OrdinalType NumDofsPerNode = EvaluationType::ElementType::mNumSpatialDims>
class ProjectedSurfaceDisplacement : 
    public AbstractSurfaceDisplacement<EvaluationType>
{
private: 
    using ElementType = typename EvaluationType::ElementType;
    using InStateT    = typename EvaluationType::StateScalarType;  
    using OutStateT   = typename EvaluationType::ResultScalarType; 

    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerFace;
    using ElementType::mNumNodesPerCell;

public:
    ProjectedSurfaceDisplacement
    (const Plato::OrdinalVectorT<Plato::OrdinalType> & aParentElements,
     const Plato::ScalarMultiVectorT<Plato::Scalar>  & aMappedLocations,
           Plato::Mesh                                 aMesh,
           Plato::Scalar                               aScale = 1.0) :
     AbstractSurfaceDisplacement<EvaluationType>(aScale),
     mParentElements(aParentElements),
     mMappedLocations(aMappedLocations),
     mGetBasis(aMesh),
     mInterpolateFromNodal(),
     mChildNode(0)
    {
    }

    KOKKOS_INLINE_FUNCTION void
    operator()
    (Plato::OrdinalType                               aCellOrdinal, 
     const Plato::Array<mNumNodesPerFace>           & aBasisFunctions,
     const Plato::ScalarMultiVectorT<InStateT>      & aState,
           Plato::Array<mNumSpatialDims, OutStateT> & aSurfaceDisp) const override
    {
        auto tParentOrdinal = aCellOrdinal*mNumNodesPerFace + mChildNode;
        auto tParentElement = mParentElements(tParentOrdinal);

        Plato::Array<mNumSpatialDims, Plato::Scalar> tInPoint(0.0);
        for(Plato::OrdinalType iDim=0; iDim<mNumSpatialDims; iDim++)
        {
            tInPoint(iDim) = mMappedLocations(iDim, tParentOrdinal);
        }

        Plato::Array<mNumNodesPerCell, Plato::Scalar> tBasis(0.0); // config scalar type
        mGetBasis(tParentElement, tInPoint, tBasis);

        mInterpolateFromNodal(tParentElement, tBasis, aState, aSurfaceDisp);

        auto tScale = this->mScale;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
        {
            aSurfaceDisp(tDofIndex) *= tScale * aBasisFunctions(mChildNode);
        }
    }

    void setChildNode(Plato::OrdinalType aChildNode)
    {
        mChildNode = aChildNode;
    }

private: 
    Plato::OrdinalVectorT<Plato::OrdinalType>             mParentElements;
    Plato::ScalarMultiVectorT<Plato::Scalar>              mMappedLocations;
    Plato::OrdinalType                                    mChildNode;
    Plato::Geometry::GetBasis<ElementType, Plato::Scalar> mGetBasis;
    Plato::InterpolateFromNodal<ElementType, NumDofsPerNode, /*offset=*/0, mNumSpatialDims> mInterpolateFromNodal;

};

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

template <typename ScalarT, typename OrdinalT>
void get_child_node_coordinates
(const Plato::ScalarVectorT<ScalarT>   & aCoords,
 const Plato::OrdinalVectorT<OrdinalT> & aNodes,
       Plato::ScalarMultiVector        & aNodeLocations)
{
    Plato::OrdinalType tNumDims = aNodeLocations.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,aNodes.size()), KOKKOS_LAMBDA(int nodeOrdinal)
    {
        auto tNodeOrdinal = aNodes(nodeOrdinal);
        for (Plato::OrdinalType iDim = 0; iDim < tNumDims; iDim++)
            aNodeLocations(iDim, nodeOrdinal) = aCoords(tNodeOrdinal*tNumDims+iDim);
    }, "get coords");

}

void map_child_nodes
(const Plato::ScalarMultiVector   & aNodeLocations,
       Plato::ScalarMultiVector   & aMappedNodeLocations,
       std::vector<Plato::Scalar> & aTranslationX,
       std::vector<Plato::Scalar> & aTranslationY,
       std::vector<Plato::Scalar> & aTranslationZ)
{
    auto transX = Plato::TestHelpers::create_device_view( aTranslationX);
    auto transY = Plato::TestHelpers::create_device_view( aTranslationY);
    auto transZ = Plato::TestHelpers::create_device_view( aTranslationZ);

    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,aNodeLocations.extent(1)), KOKKOS_LAMBDA(int nodeOrdinal)
    {
        aMappedNodeLocations(0,nodeOrdinal) = aNodeLocations(0,nodeOrdinal) + transX(nodeOrdinal);
        aMappedNodeLocations(1,nodeOrdinal) = aNodeLocations(1,nodeOrdinal) + transY(nodeOrdinal);
        aMappedNodeLocations(2,nodeOrdinal) = aNodeLocations(2,nodeOrdinal) + transZ(nodeOrdinal);
    }, "get coords");
}

TEUCHOS_UNIT_TEST(ParsingTests, ParseSingleContactPair)
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

TEUCHOS_UNIT_TEST(ContactPairTests, ComputeAndAccessChildNodesAndParentElements)
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

    auto tContactParams = tInputs->sublist("Contact");
    auto tPairsParams = tContactParams.sublist("Pairs");
    const auto& tMyName = tPairsParams.name(tPairsParams.begin());
    Teuchos::ParameterList& tPairParams = tPairsParams.sublist(tMyName);

    Plato::ContactPair tPair = Plato::parseContactPair(tPairParams, tMesh);

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

    // test parent elements
    auto tChildLocations       = Plato::computeNodeLocations(tMesh, tPair.childNodesA);
    auto tMappedChildLocations = Plato::mapNodeLocations(tChildLocations, tPair.initialGap);
    Plato::SpatialDomain tDomain = Plato::getDomain(tPair.parentBlockB, tSpatialModel.Domains);

    Plato::OrdinalVector tParentElementsA("parent elements", tPair.childNodesA.size());
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
      (tMesh, tDomain.cellOrdinals(), tChildLocations, tMappedChildLocations, tParentElementsA);

    auto tParentElements_Host = Plato::TestHelpers::get( tParentElementsA );
    std::vector<Plato::OrdinalType> tParentElements_Gold = {7, 6, 6, 6};
    for(int iVal=0; iVal<tParentElements_Gold.size(); iVal++){
        TEST_EQUALITY(tParentElements_Host(iVal), tParentElements_Gold[iVal]);
    }

    tChildLocations       = Plato::computeNodeLocations(tMesh, tPair.childNodesB);
    tMappedChildLocations = Plato::mapNodeLocations(tChildLocations, tPair.initialGap, -1.0);
    tDomain = Plato::getDomain(tPair.parentBlockA, tSpatialModel.Domains);

    Plato::OrdinalVector tParentElementsB("parent elements", tPair.childNodesB.size());
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
      (tMesh, tDomain.cellOrdinals(), tChildLocations, tMappedChildLocations, tParentElementsB);

    tParentElements_Host = Plato::TestHelpers::get( tParentElementsB );
    tParentElements_Gold = {4, 2, 2, 0};
    for(int iVal=0; iVal<tParentElements_Gold.size(); iVal++){
        TEST_EQUALITY(tParentElements_Host(iVal), tParentElements_Gold[iVal]);
    }
}

TEUCHOS_UNIT_TEST(ProjectionTests, FindParentElementsForNodesWithDifferentTranslations)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
    constexpr int tSpaceDim = ElementType::mNumSpatialDims;

    Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);

    auto tNodesXMinus = tMesh->GetNodeSetNodes("x-");
    auto tNumberChildNodes = tNodesXMinus.size();

    // map child face coordinates
    Plato::ScalarMultiVector tChildNodeLocations       ("child node locations", tSpaceDim, tNumberChildNodes);
    get_child_node_coordinates(tMesh->Coordinates(),tNodesXMinus,tChildNodeLocations);

    std::vector<Plato::Scalar> tTranslationsX = {0.25, 1.0, 0.5, 0.5};
    std::vector<Plato::Scalar> tTranslationsY = {0.5, 0.5, -1.0, -1.0};
    std::vector<Plato::Scalar> tTranslationsZ = {1.0, -0.75, 0.25, -0.25};

    Plato::ScalarMultiVector tMappedChildNodeLocations ("mapped child node locations", tSpaceDim, tNumberChildNodes);
    map_child_nodes(tChildNodeLocations,tMappedChildNodeLocations,tTranslationsX,tTranslationsY,tTranslationsZ);

    // find parent elements
    Plato::OrdinalVector tParentElements("mapped elements", tNumberChildNodes);

    auto tDomainCellMap = tSpatialModel.Domains.front().cellOrdinals(); // first and only domain
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
      (tSpatialModel.Mesh, tDomainCellMap, tChildNodeLocations, tMappedChildNodeLocations, tParentElements);

    // test parent elements
    auto tParentElements_Host = Plato::TestHelpers::get( tParentElements );
    std::vector<Plato::OrdinalType> tParentElements_gold = { 2, 5, 4, 3 };

    for(int iParent=0; iParent<int(tParentElements_gold.size()); iParent++){
        TEST_EQUALITY(tParentElements_Host(iParent), tParentElements_gold[iParent]);
    }
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_ChildElementContrbution)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    // create dummy displacement workset from box mesh
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);
     
    // get basis functions at cubature (quadrature, since surface) point
    Plato::OrdinalType tCubOrdinal = 0;
    auto tCubPoints = ElementType::Face::getCubPoints();
    auto tCubPoint = tCubPoints(tCubOrdinal);
    auto tBasisValues = ElementType::Face::basisValues(tCubPoint);

    // get side set info
    std::string tSideSetName = "z-";
    auto tChildFaceElements = tMesh->GetSideSetElements(tSideSetName);
    auto tNumChildCells = tChildFaceElements.extent(0);
    auto tChildFaceLocalNodes = tMesh->GetSideSetLocalNodes(tSideSetName);

    // construct dummy residual class
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    DummyResidual<EvaluationType> tResidual;

    // construct compute surface displacement functor
    SurfaceDisplacement<EvaluationType> tComputeSurfaceDisp(tChildFaceElements, tChildFaceLocalNodes, -1.0);

    // test surface displacement child face cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;
    Plato::ScalarVector tSurfaceDisp0("make on device", ElementType::mNumDofsPerNode);
    tResidual.exercise_surface_disp_interface(tComputeSurfaceDisp, tChildCellOrdinal, tBasisValues, tDispWS, tSurfaceDisp0);

    auto tSurfaceDisp0_Host = Plato::TestHelpers::get( tSurfaceDisp0 );
    std::vector<double> tSurfaceDisp_Gold = {-0.0009, -0.0010, -0.0011};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp0_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }

    // test surface displacement child face cell 1
    tChildCellOrdinal = 1;
    Plato::ScalarVector tSurfaceDisp1("make on device", ElementType::mNumDofsPerNode);
    tResidual.exercise_surface_disp_interface(tComputeSurfaceDisp, tChildCellOrdinal, tBasisValues, tDispWS, tSurfaceDisp1);

    auto tSurfaceDisp1_Host = Plato::TestHelpers::get( tSurfaceDisp1 );
    tSurfaceDisp_Gold = {-0.0011, -0.0012, -0.0013};
    for(int iDof=0; iDof<tSurfaceDisp_Gold.size(); iDof++){
        TEST_FLOATING_EQUALITY(tSurfaceDisp1_Host(iDof), tSurfaceDisp_Gold[iDof], 1e-12);
    }
}

TEUCHOS_UNIT_TEST(FunctorTests, SurfaceDisplacement_SingleParentElementContribution)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
    // create dummy displacement workset from box mesh
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // get side set info
    std::string tSideSetName = "z-";
    auto tChildFaceElements = tMesh->GetSideSetElements(tSideSetName);
    auto tNumChildCells = tChildFaceElements.extent(0);
    auto tChildFaceLocalNodes = tMesh->GetSideSetLocalNodes(tSideSetName);

    // get list of all child element nodes on face in order of element ID (repeated values b/c of shared nodes, but is ok) 
    Plato::OrdinalVector tChildFaceElementNodeIDs("ids of child face element nodes", tNumChildCells*ElementType::mNumNodesPerFace);
    auto tCells2Nodes = tMesh->Connectivity();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumChildCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tChildFaceElements(aCellOrdinal);

        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerFace; tNodeIndex++)
        {
            auto tLocalNodeOrdinal = tChildFaceLocalNodes(aCellOrdinal*ElementType::mNumNodesPerFace+tNodeIndex);
            auto tGlobalNodeOrdinal = tCells2Nodes(tCellOrdinal*ElementType::mNumNodesPerCell + tLocalNodeOrdinal);
            tChildFaceElementNodeIDs(aCellOrdinal*ElementType::mNumNodesPerFace + tNodeIndex) = tGlobalNodeOrdinal;
        }
    }, "get parent elements for local ele face nodes");

    // map child face coordinates - setting manually
    Plato::ScalarMultiVector tChildElementNodeLocations("child node locations", ElementType::mNumSpatialDims, tChildFaceElementNodeIDs.size());
    get_child_node_coordinates(tMesh->Coordinates(),tChildFaceElementNodeIDs,tChildElementNodeLocations);

    std::vector<Plato::Scalar> tTranslationsX = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<Plato::Scalar> tTranslationsY = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<Plato::Scalar> tTranslationsZ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    Plato::ScalarMultiVector tMappedChildElementNodeLocations ("mapped child node locations", ElementType::mNumSpatialDims, tChildFaceElementNodeIDs.size());
    map_child_nodes(tChildElementNodeLocations,tMappedChildElementNodeLocations,tTranslationsX,tTranslationsY,tTranslationsZ);

    // find parent elements
    Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);
    auto tDomainCellMap = tSpatialModel.Domains.front().cellOrdinals(); // first and only domain
    Plato::OrdinalVector tParentElements("mapped child face element nodes", tNumChildCells*ElementType::mNumNodesPerFace);
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
      (tSpatialModel.Mesh, tDomainCellMap, tChildElementNodeLocations, tMappedChildElementNodeLocations, tParentElements);
     
    // get basis functions at cubature (quadrature, since surface) point
    Plato::OrdinalType tCubOrdinal = 0;
    auto tCubPoints = ElementType::Face::getCubPoints();
    auto tCubPoint = tCubPoints(tCubOrdinal);
    auto tBasisValues = ElementType::Face::basisValues(tCubPoint);

    // construct dummy residual class
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    DummyResidual<EvaluationType> tResidual;

    // construct compute surface displacement functor
    ProjectedSurfaceDisplacement<EvaluationType> tComputeSurfaceDisp(tParentElements, tMappedChildElementNodeLocations, tMesh);

    // test surface displacement terms for each child node on child cell 0
    Plato::OrdinalType tChildCellOrdinal = 0;

    std::vector<std::vector<double>> tSurfaceDisp_Gold = {
        {0.0004 / 3.0, 0.0005 / 3.0, 0.0006 / 3.0},
        {0.0010 / 3.0, 0.0011 / 3.0, 0.0012 / 3.0},
        {0.0022 / 3.0, 0.0023 / 3.0, 0.0024 / 3.0}
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
        {0.0004 / 3.0, 0.0005 / 3.0, 0.0006 / 3.0},
        {0.0022 / 3.0, 0.0023 / 3.0, 0.0024 / 3.0},
        {0.0016 / 3.0, 0.0017 / 3.0, 0.0018 / 3.0}
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
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
    // create dummy displacement workset from box mesh
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    std::vector<Plato::Scalar> u_host( ElementType::mNumSpatialDims*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tMesh->NumElements(), ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // get side set info
    std::string tSideSetName = "z-";
    auto tChildFaceElements = tMesh->GetSideSetElements(tSideSetName);
    auto tNumChildCells = tChildFaceElements.extent(0);
    auto tChildFaceLocalNodes = tMesh->GetSideSetLocalNodes(tSideSetName);

    // get list of all child element nodes on face in order of element ID (repeated values b/c of shared nodes, but is ok) 
    Plato::OrdinalVector tChildFaceElementNodeIDs("ids of child face element nodes", tNumChildCells*ElementType::mNumNodesPerFace);
    auto tCells2Nodes = tMesh->Connectivity();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumChildCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tChildFaceElements(aCellOrdinal);

        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerFace; tNodeIndex++)
        {
            auto tLocalNodeOrdinal = tChildFaceLocalNodes(aCellOrdinal*ElementType::mNumNodesPerFace+tNodeIndex);
            auto tGlobalNodeOrdinal = tCells2Nodes(tCellOrdinal*ElementType::mNumNodesPerCell + tLocalNodeOrdinal);
            tChildFaceElementNodeIDs(aCellOrdinal*ElementType::mNumNodesPerFace + tNodeIndex) = tGlobalNodeOrdinal;
        }
    }, "get parent elements for local ele face nodes");

    // map child face coordinates - setting manually
    Plato::ScalarMultiVector tChildElementNodeLocations("child node locations", ElementType::mNumSpatialDims, tChildFaceElementNodeIDs.size());
    get_child_node_coordinates(tMesh->Coordinates(),tChildFaceElementNodeIDs,tChildElementNodeLocations);

    std::vector<Plato::Scalar> tTranslationsX = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<Plato::Scalar> tTranslationsY = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::vector<Plato::Scalar> tTranslationsZ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    Plato::ScalarMultiVector tMappedChildElementNodeLocations ("mapped child node locations", ElementType::mNumSpatialDims, tChildFaceElementNodeIDs.size());
    map_child_nodes(tChildElementNodeLocations,tMappedChildElementNodeLocations,tTranslationsX,tTranslationsY,tTranslationsZ);

    // find parent elements
    Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);
    auto tDomainCellMap = tSpatialModel.Domains.front().cellOrdinals(); // first and only domain
    Plato::OrdinalVector tParentElements("mapped child face element nodes", tNumChildCells*ElementType::mNumNodesPerFace);
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
      (tSpatialModel.Mesh, tDomainCellMap, tChildElementNodeLocations, tMappedChildElementNodeLocations, tParentElements);
     
    // construct dummy residual class
    using EvaluationType = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    DummyResidual<EvaluationType> tResidual;

    // construct compute surface displacement functors
    SurfaceDisplacement<EvaluationType> computeChildSurfaceDisp(tChildFaceElements, tChildFaceLocalNodes, -1.0);
    ProjectedSurfaceDisplacement<EvaluationType> computeParentSurfaceDisp(tParentElements, tMappedChildElementNodeLocations, tMesh);

    // test computation of displacement difference (dummy contact force)
    Plato::ScalarMultiVectorT<Plato::Scalar> tResult("dummy contact force", tNumChildCells, ElementType::mNumDofsPerCell);
    tResidual.dummy_contact_force(tSpatialModel,tSideSetName,tDispWS,computeChildSurfaceDisp,tResult); // child face contributions

    for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
    {
        computeParentSurfaceDisp.setChildNode(iChildNode);
        tResidual.dummy_contact_force(tSpatialModel,tSideSetName,tDispWS,computeParentSurfaceDisp,tResult); // parent face contributions
    }

    std::vector<std::vector<double>> tResult_Gold = {
        {0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0},
        {0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0, 0.0, 0.0}
    };

    auto tResult_Host = Plato::TestHelpers::get( tResult );

    for(int iCell=0; iCell<int(tNumChildCells); iCell++){
        for(int iDof=0; iDof<ElementType::mNumNodesPerFace*ElementType::mNumDofsPerNode; iDof++){
            TEST_FLOATING_EQUALITY(tResult_Host(iCell,iDof), tResult_Gold[iCell][iDof], 1e-12);
      }
    }

}

// TEUCHOS_UNIT_TEST(WorksetTests, WorksetFullDisplacements_TreatProjectedDisplacementsAsFullWorkset)
// {
//     //*********************************************************************//
//     // THIS TEST ALSO checks that displacement values are projected correctly
//     // which may be useful to keep even if worksetting is done differently...
//     //*********************************************************************//

//     constexpr Plato::OrdinalType tMeshWidth = 1;
//     auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

//     using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
//     constexpr int tSpaceDim = ElementType::mNumSpatialDims;
//     int tNumCells = tMesh->NumElements();
//     constexpr int tDofsPerCell = ElementType::mNumDofsPerCell;
//     constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
//     constexpr int tNumNodesPerFace  = ElementType::mNumNodesPerFace;

//     Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);
//     Plato::VectorEntryOrdinal<tSpaceDim, ElementType::mNumDofsPerNode, ElementType::mNumNodesPerCell> tStateEntryOrdinal(tMesh);

//     // create mesh based displacement from host data
//     //
//     std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
//     Plato::Scalar disp = 0.0, dval = 0.0001;
//     for( auto& val : u_host ) val = (disp += dval);
//     Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//       u_host_view(u_host.data(),u_host.size());
//     auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

//     Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
//     Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tNumCells, tDofsPerCell);
//     tWorksetBase.worksetState(u, tDispWS);

//     // get side set info
//     //
//     std::string tSideSetName = "z-";
//     auto tChildFaceNodes = tMesh->GetNodeSetNodes(tSideSetName);
//     auto tNumChildNodes = tChildFaceNodes.extent(0);
//     auto tChildFaceElements = tMesh->GetSideSetElements(tSideSetName);
//     auto tNumChildCells = tChildFaceElements.extent(0);
//     auto tChildFaceOrdinals = tMesh->GetSideSetFaces(tSideSetName);
//     auto tChildFaceLocalNodes = tMesh->GetSideSetLocalNodes(tSideSetName);

//     // setting manually, but would need to read translation, or compute initial gap
//     Plato::ScalarMultiVector tChildNodeLocations       ("child node locations",        tSpaceDim, tNumChildNodes);
//     Plato::ScalarMultiVector tMappedChildNodeLocations ("mapped child node locations", tSpaceDim, tNumChildNodes);

//     auto coords = tMesh->Coordinates();
//     Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumChildNodes), KOKKOS_LAMBDA(int nodeOrdinal)
//     {
//       auto tNodeOrdinal = tChildFaceNodes(nodeOrdinal);
//       tChildNodeLocations(0, nodeOrdinal) = coords(tNodeOrdinal*tSpaceDim+0);
//       tChildNodeLocations(1, nodeOrdinal) = coords(tNodeOrdinal*tSpaceDim+1);
//       tChildNodeLocations(2, nodeOrdinal) = coords(tNodeOrdinal*tSpaceDim+2);

//       tMappedChildNodeLocations(0, nodeOrdinal) = coords(tNodeOrdinal*tSpaceDim+0);
//       tMappedChildNodeLocations(1, nodeOrdinal) = coords(tNodeOrdinal*tSpaceDim+1);
//       tMappedChildNodeLocations(2, nodeOrdinal) = coords(tNodeOrdinal*tSpaceDim+2) + 1.0;
//     }, "get coords");

//     // find parent elements
//     auto tDomainCellMap = tSpatialModel.Domains.front().cellOrdinals(); // first and only domain
//     Plato::OrdinalVector tParentElements("mapped elements", tNumChildNodes);
//     Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
//       (tSpatialModel.Mesh, tDomainCellMap, tChildNodeLocations, tMappedChildNodeLocations, tParentElements);

//     // project displacements
//     Plato::Geometry::GetBasis<ElementType, Plato::Scalar> getBasis(tMesh);
//     Plato::InterpolateFromNodal<ElementType, tNumDofsPerNode, /*offset=*/0, tSpaceDim> interpolateFromNodal;

//     // to get the state entry ordinal correct, this has to be the full displacement
//     // so I have to copy the whole displacement field to only change a few entries
//     // it seems like a waste of memory
//     Plato::ScalarVector tProjectedDisp("projected displacement", u.extent(0));
//     Kokkos::deep_copy(tProjectedDisp, u); // this can just be initialized as 0

//     Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType iChildNode)
//     {
//         auto tChildNode = tChildFaceNodes(iChildNode);
//         auto tParentElement = tParentElements(iChildNode);
        
//         Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);
//         for(Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
//         {
//             tInPoint(iDim) = tMappedChildNodeLocations(iDim, iChildNode);
//         }

//         Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tBasis(0.0); // config scalar type
//         getBasis(tParentElement, tInPoint, tBasis);

//         Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tProjectedDisplacement(0.0);
//         interpolateFromNodal(tParentElement, tBasis, tDispWS, tProjectedDisplacement);

//         for(Plato::OrdinalType iDof=0; iDof<ElementType::mNumDofsPerNode; iDof++)
//         {
//             tProjectedDisp(tChildNode*ElementType::mNumDofsPerNode + iDof) = tProjectedDisplacement(iDof);
//         }
        
//     }, "get displacement values of parent face at child node locations");

//     // workset projected displacements
//     Plato::ScalarMultiVectorT<Plato::Scalar> tProjectedDispWS("state workset", tNumChildCells, ElementType::mNumNodesPerFace*ElementType::mNumDofsPerNode);

//     // here will I need to workset for Fad type correctly? Or since tProjectedDisp is computed from 
//     // tStateWS is it okay?
//     Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumChildCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
//     {
//         auto tCellOrdinal = tChildFaceElements(aCellOrdinal);

//         for(Plato::OrdinalType tDofIndex = 0; tDofIndex < ElementType::mNumDofsPerNode; tDofIndex++)
//         {
//             for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerFace; tNodeIndex++)
//             {
//                 auto tLocalNodeOrdinal = tChildFaceLocalNodes(aCellOrdinal*ElementType::mNumNodesPerFace+tNodeIndex);

//                 Plato::OrdinalType tEntryOrdinal = tStateEntryOrdinal(tCellOrdinal, tLocalNodeOrdinal, tDofIndex);
//                 Plato::OrdinalType tLocalDof = tNodeIndex * ElementType::mNumDofsPerNode + tDofIndex;
//                 tProjectedDispWS(aCellOrdinal, tLocalDof) = tProjectedDisp(tEntryOrdinal);
//             }
//         }
//     }, "workset_state_scalar_scalar");

//     // TEST workset came out as expected
//     //
//     std::vector<std::vector<Plato::Scalar>> tProjectedDispWS_gold = { 
//     { 0.0004, 0.0005, 0.0006, 0.0010, 0.0011, 0.0012, 0.0022, 0.0023, 0.0024 },
//     { 0.0004, 0.0005, 0.0006, 0.0022, 0.0023, 0.0024, 0.0016, 0.0017, 0.0018 }
//     };

//     auto tProjectedDispWS_Host = Plato::TestHelpers::get( tProjectedDispWS );

//     for(int iCell=0; iCell<int(tNumChildCells); iCell++){
//         for(int iDof=0; iDof<tNumNodesPerFace*tNumDofsPerNode; iDof++){
//             TEST_FLOATING_EQUALITY(tProjectedDispWS_Host(iCell,iDof), tProjectedDispWS_gold[iCell][iDof], 1e-12);
//       }
//     }

//     // TEST that displacement difference between projected and child nodes
//     // is as expected, i.e. check that worksets line up
//     // all projected elements are 1 more than child, meaning they have displacement
//     // values 3 higher
//     //
//     std::vector<std::vector<Plato::Scalar>> tGoldChildMinusProjected = { 
//     { 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003 },
//     { 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003 }
//     };

//     auto tDispWS_Host = Plato::TestHelpers::get( tDispWS );

//     auto tChildFaceElements_Host = Plato::TestHelpers::get( tChildFaceElements );
//     auto tChildFaceLocalNodes_Host = Plato::TestHelpers::get( tChildFaceLocalNodes );

//     for(int iCell=0; iCell<int(tNumChildCells); iCell++){
//         auto tCellOrdinal = tChildFaceElements_Host(iCell);
//         for(int iNode=0; iNode<tNumNodesPerFace; iNode++){
//             auto tFullEleWSNodeOrdinal = tChildFaceLocalNodes_Host(iCell*tNumNodesPerFace+iNode);
//             for(int iDof=0; iDof<tNumDofsPerNode; iDof++){
//                 auto tProjectedEleWSDof = iNode*tNumDofsPerNode + iDof;
//                 auto tFullEleWSDof = tFullEleWSNodeOrdinal*tNumDofsPerNode + iDof;
//                 auto tDiff = tProjectedDispWS_Host(iCell,tProjectedEleWSDof) - tDispWS_Host(tCellOrdinal,tFullEleWSDof);
//                 TEST_FLOATING_EQUALITY(tDiff, tGoldChildMinusProjected[iCell][tProjectedEleWSDof], 1e-12);
//             }
//         }
//     }
// }

// TEUCHOS_UNIT_TEST(WorksetTests, WorksetChildDisplacements_WorksetParentElements)
// {
//     constexpr Plato::OrdinalType tMeshWidth = 1;
//     auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

//     using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
//     constexpr int tSpaceDim = ElementType::mNumSpatialDims;
//     int tNumCells = tMesh->NumElements();
//     constexpr int tDofsPerCell = ElementType::mNumDofsPerCell;
//     constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;

//     Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);
//     Plato::VectorEntryOrdinal<ElementType::mNumSpatialDims, ElementType::mNumDofsPerNode, ElementType::mNumNodesPerCell> tStateEntryOrdinal(tMesh);

//     // create mesh based displacement from host data
//     //
//     std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
//     Plato::Scalar disp = 0.0, dval = 0.0001;
//     for( auto& val : u_host ) val = (disp += dval);
//     Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//       u_host_view(u_host.data(),u_host.size());
//     auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

//     Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
//     Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tNumCells, tDofsPerCell);
//     tWorksetBase.worksetState(u, tDispWS);

//     // get side set info
//     //
//     std::string tSideSetName = "z-";
//     auto tChildFaceNodes = tMesh->GetNodeSetNodes(tSideSetName);
//     auto tNumChildNodes = tChildFaceNodes.extent(0);
//     auto tChildFaceElements = tMesh->GetSideSetElements(tSideSetName);
//     auto tNumChildCells = tChildFaceElements.extent(0);
//     auto tChildFaceOrdinals = tMesh->GetSideSetFaces(tSideSetName);
//     auto tChildFaceLocalNodes = tMesh->GetSideSetLocalNodes(tSideSetName);

//     // map child face coordinates - setting manually
//     Plato::ScalarMultiVector tChildNodeLocations("child node locations", tSpaceDim, tNumChildNodes);
//     get_child_node_coordinates(tMesh->Coordinates(),tChildFaceNodes,tChildNodeLocations);

//     std::vector<Plato::Scalar> tTranslationsX = {0.5, 0.5, -0.5, -0.5};
//     std::vector<Plato::Scalar> tTranslationsY = {0.25, -0.25, 0.25, -0.25};
//     std::vector<Plato::Scalar> tTranslationsZ = {1.0, 1.0, 1.0, 1.0};

//     Plato::ScalarMultiVector tMappedChildNodeLocations ("mapped child node locations", tSpaceDim, tNumChildNodes);
//     map_child_nodes(tChildNodeLocations,tMappedChildNodeLocations,tTranslationsX,tTranslationsY,tTranslationsZ);

//     // find parent elements
//     auto tDomainCellMap = tSpatialModel.Domains.front().cellOrdinals(); // first and only domain
//     Plato::OrdinalVector tParentElements("mapped elements", tNumChildNodes);
//     Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
//       (tSpatialModel.Mesh, tDomainCellMap, tChildNodeLocations, tMappedChildNodeLocations, tParentElements);

//     // workset state for child face elements
//     auto tElements = tChildFaceElements; // calling it something generic to pull this out as a function later
//     auto tState = u; // calling it something generic to pull this out as a function later
//     auto tNumNodesPerCell = ElementType::mNumNodesPerCell; // calling it something generic to pull this out as a function later

//     Plato::ScalarMultiVectorT<Plato::Scalar> tChildFaceDispWS("child face state workset", tNumChildCells, ElementType::mNumNodesPerCell*ElementType::mNumDofsPerNode);
//     Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tElements.size()), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
//     {
//         auto tCellOrdinal = tElements(aCellOrdinal);
//         for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
//         {
//             for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
//             {
//                 Plato::OrdinalType tEntryOrdinal = tStateEntryOrdinal(tCellOrdinal, tNodeIndex, tDofIndex);
//                 Plato::OrdinalType tLocalDof = (tNodeIndex * tNumDofsPerNode) + tDofIndex;
//                 tChildFaceDispWS(aCellOrdinal, tLocalDof) = tState(tEntryOrdinal);
//             }
//         }
//     }, "workset child face elements state");

//     // TEST child face element workset came out as expected
//     //
//     std::vector<std::vector<Plato::Scalar>> tChildFaceDispWS_gold = { 
//     { 0.0001, 0.0002, 0.0003, 0.0019, 0.0020, 0.0021, 0.0007, 0.0008, 0.0009, 0.0022, 0.0023, 0.0024 },
//     { 0.0001, 0.0002, 0.0003, 0.0013, 0.0014, 0.0015, 0.0019, 0.0020, 0.0021, 0.0022, 0.0023, 0.0024 },
//     };

//     auto tChildFaceDispWS_Host = Plato::TestHelpers::get( tChildFaceDispWS );

//     for(int iCell=0; iCell<int(tNumChildCells); iCell++){
//         for(int iDof=0; iDof<tNumNodesPerCell*tNumDofsPerNode; iDof++){
//             TEST_FLOATING_EQUALITY(tChildFaceDispWS_Host(iCell,iDof), tChildFaceDispWS_gold[iCell][iDof], 1e-12);
//       }
//     }

// }

// TEUCHOS_UNIT_TEST(AssemblyTests, ComputeResidualTerms)
// {
//     constexpr Plato::OrdinalType tMeshWidth = 1;
//     auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

//     using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
//     constexpr int tSpaceDim = ElementType::mNumSpatialDims;
//     int tNumCells = tMesh->NumElements();
//     constexpr int tDofsPerCell = ElementType::mNumDofsPerCell;
//     constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
//     constexpr int tNumNodesPerFace  = ElementType::mNumNodesPerFace;
//     constexpr int tNumNodesPerCell  = ElementType::mNumNodesPerCell;

//     Plato::SpatialModel tSpatialModel = setup_dummy_spatial_model(tMesh);
//     Plato::VectorEntryOrdinal<ElementType::mNumSpatialDims, ElementType::mNumDofsPerNode, ElementType::mNumNodesPerCell> tStateEntryOrdinal(tMesh);

//     // create mesh based displacement from host data
//     //
//     std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
//     Plato::Scalar disp = 0.0, dval = 0.0001;
//     for( auto& val : u_host ) val = (disp += dval);
//     Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
//       u_host_view(u_host.data(),u_host.size());
//     auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

//     // workset displacement
//     Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
//     Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tNumCells, tDofsPerCell);
//     tWorksetBase.worksetState(u, tDispWS);

//     // workset config
//     Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("Config Workset", tNumCells, tNumNodesPerCell, tSpaceDim);
//     tWorksetBase.worksetConfig(tConfigWS);

//     // get side set info
//     //
//     std::string tSideSetName = "z-";
//     auto tChildFaceNodes = tMesh->GetNodeSetNodes(tSideSetName);
//     auto tNumChildNodes = tChildFaceNodes.extent(0);
//     auto tChildFaceElements = tMesh->GetSideSetElements(tSideSetName);
//     auto tNumChildCells = tChildFaceElements.extent(0);
//     auto tChildFaceOrdinals = tMesh->GetSideSetFaces(tSideSetName);
//     auto tChildFaceLocalNodes = tMesh->GetSideSetLocalNodes(tSideSetName);

//     // get list of all child element nodes on face in order of element ID (repeated values b/c of shared nodes, but is ok) 
//     Plato::OrdinalVector tChildFaceElementNodeIDs("ids of child face element nodes", tNumChildCells*tNumNodesPerFace);
//     auto tCells2Nodes = tMesh->Connectivity();
//     Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumChildCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
//     {
//         auto tCellOrdinal = tChildFaceElements(aCellOrdinal);

//         for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerFace; tNodeIndex++)
//         {
//             auto tLocalNodeOrdinal = tChildFaceLocalNodes(aCellOrdinal*ElementType::mNumNodesPerFace+tNodeIndex);
//             auto tGlobalNodeOrdinal = tCells2Nodes(tCellOrdinal*ElementType::mNumNodesPerCell + tLocalNodeOrdinal);
//             tChildFaceElementNodeIDs(aCellOrdinal*tNumNodesPerFace + tNodeIndex) = tGlobalNodeOrdinal;
//         }
//     }, "get parent elements for local ele face nodes");

//     // check that child element nodes are stored correctly
//     std::vector<Plato::OrdinalType> tChildFaceElementNodeIDs_Gold = {0, 2, 6, 0, 6, 4};
//     auto tChildFaceElementNodeIDs_Host = Plato::TestHelpers::get( tChildFaceElementNodeIDs );

//     for(int iOrdinal=0; iOrdinal<tChildFaceElementNodeIDs_Gold.size(); iOrdinal++){
//         TEST_EQUALITY(tChildFaceElementNodeIDs_Host(iOrdinal), tChildFaceElementNodeIDs_Gold[iOrdinal]);
//     }

//     // map child face coordinates - setting manually
//     Plato::ScalarMultiVector tChildElementNodeLocations("child node locations", tSpaceDim, tChildFaceElementNodeIDs.size());
//     get_child_node_coordinates(tMesh->Coordinates(),tChildFaceElementNodeIDs,tChildElementNodeLocations);

//     // std::vector<Plato::Scalar> tTranslationsX = {0.5, 0.5, -0.5, 0.5, -0.5, -0.5};
//     // std::vector<Plato::Scalar> tTranslationsY = {0.25, -0.25, -0.25, 0.25, -0.25, 0.25};
//     // std::vector<Plato::Scalar> tTranslationsZ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
//     std::vector<Plato::Scalar> tTranslationsX = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//     std::vector<Plato::Scalar> tTranslationsY = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//     std::vector<Plato::Scalar> tTranslationsZ = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

//     Plato::ScalarMultiVector tMappedChildElementNodeLocations ("mapped child node locations", tSpaceDim, tChildFaceElementNodeIDs.size());
//     map_child_nodes(tChildElementNodeLocations,tMappedChildElementNodeLocations,tTranslationsX,tTranslationsY,tTranslationsZ);

//     // find parent elements
//     auto tDomainCellMap = tSpatialModel.Domains.front().cellOrdinals(); // first and only domain
//     Plato::OrdinalVector tParentElements("mapped child face element nodes", tNumChildCells*tNumNodesPerFace);
//     Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
//       (tSpatialModel.Mesh, tDomainCellMap, tChildElementNodeLocations, tMappedChildElementNodeLocations, tParentElements);

//     // compute contact force
//     Plato::Geometry::GetBasis<ElementType, Plato::Scalar> getBasis(tMesh);
//     Plato::InterpolateFromNodal<ElementType, tNumDofsPerNode, /*offset=*/0, tSpaceDim> interpolateFromNodal;
//     Plato::WeightedNormalVector<ElementType> weightedNormalVector;
//     Plato::SurfaceArea<ElementType> surfaceArea;

//     auto tCubaturePoints  = ElementType::Face::getCubPoints();
//     auto tCubatureWeights = ElementType::Face::getCubWeights();
//     auto tNumPoints = tCubatureWeights.size();

//     Plato::ScalarMultiVectorT<Plato::Scalar> tProjectedDispWS("state workset", tNumChildCells, ElementType::mNumNodesPerFace*ElementType::mNumDofsPerNode);

//     Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumChildCells, tNumPoints}),
//     KOKKOS_LAMBDA(const Plato::OrdinalType aCellOrdinal, const Plato::OrdinalType aGPOrdinal)
//     {
//         auto tCellOrdinal = tChildFaceElements(aCellOrdinal);

//         Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
//         for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerFace; tNodeIndex++)
//         {
//             tLocalNodeOrds(tNodeIndex) = tChildFaceLocalNodes(aCellOrdinal*ElementType::mNumNodesPerFace+tNodeIndex);
//         }

//         // element basis gradients
//         auto tCubaturePoint = tCubaturePoints(aGPOrdinal);
//         auto tBasisGrads = ElementType::Face::basisGrads(tCubaturePoint);

//         // compute normal 
//         Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tWeightedNormalVec;
//         weightedNormalVector(tCellOrdinal, tLocalNodeOrds, tBasisGrads, tConfigWS, tWeightedNormalVec);

//         // compute surface area 
//         Plato::Scalar tSurfaceArea(0.0);
//         surfaceArea(tCellOrdinal, tLocalNodeOrds, tBasisGrads, tConfigWS, tSurfaceArea);
        
//         // compute surface area for interpolation (is in weighted normal?)

//         for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerFace; tNodeIndex++)
//         {
//             // interpolate displacement at projected child node
//             auto tParentOrdinal = aCellOrdinal*ElementType::mNumNodesPerFace + tNodeIndex;
//             auto tParentElement = tParentElements(tParentOrdinal);

//             Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);
//             for(Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
//             {
//                 tInPoint(iDim) = tMappedChildElementNodeLocations(iDim, tParentOrdinal);
//             }
//             printf("\n chile ele %d node %d has parent element %d and coordinates (%e, %e, %e) \n", aCellOrdinal, tNodeIndex, tParentElement, tInPoint(0), tInPoint(1), tInPoint(2));

//             Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tBasis(0.0); // config scalar type
//             getBasis(tParentElement, tInPoint, tBasis);

//             Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tProjectedDisplacement(0.0);
//             interpolateFromNodal(tParentElement, tBasis, tDispWS, tProjectedDisplacement);

//             // store interpolated displacement
//             for(Plato::OrdinalType tDofIndex = 0; tDofIndex < ElementType::mNumDofsPerNode; tDofIndex++)
//             {
//                 Plato::OrdinalType tLocalDof = tNodeIndex * ElementType::mNumDofsPerNode + tDofIndex;
//                 tProjectedDispWS(aCellOrdinal, tLocalDof) = tProjectedDisplacement(tDofIndex);
//             }
//         }
            
//     });

//     // TEST projected displacement values came out as expected
//     //
//     std::vector<std::vector<Plato::Scalar>> tProjectedDispWS_gold = { 
//     { 0.0004, 0.0005, 0.0006, 0.0010, 0.0011, 0.0012, 0.0022, 0.0023, 0.0024 },
//     { 0.0004, 0.0005, 0.0006, 0.0022, 0.0023, 0.0024, 0.0016, 0.0017, 0.0018 }
//     };

//     auto tProjectedDispWS_Host = Plato::TestHelpers::get( tProjectedDispWS );

//     for(int iCell=0; iCell<int(tNumChildCells); iCell++){
//         for(int iDof=0; iDof<tNumNodesPerFace*tNumDofsPerNode; iDof++){
//             TEST_FLOATING_EQUALITY(tProjectedDispWS_Host(iCell,iDof), tProjectedDispWS_gold[iCell][iDof], 1e-12);
//       }
//     }

// }

}

