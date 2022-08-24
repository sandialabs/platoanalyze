#include "PlatoTestHelpers.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <sstream>
#include <fstream>
#include <stdio.h>

#include "PlatoStaticsTypes.hpp"

#include "Tet4.hpp"
#include "MechanicsElement.hpp"
// #include "ElementBase.hpp"

#include "Plato_InputData.hpp"
#include "Plato_Exceptions.hpp"
#include "Plato_Parser.hpp"
#include "Plato_MeshMap.hpp"

#include "WorksetBase.hpp"
#include "SpatialModel.hpp"

#include "InterpolateFromNodal.hpp"

namespace ContactTests
{

TEUCHOS_UNIT_TEST(ProjectionTests, FindParentElementsForNodesWithDifferentTranslations)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
    constexpr int tSpaceDim = ElementType::mNumSpatialDims;

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

    Plato::SpatialModel tSpatialModel(tMesh, *tInputs);

    auto tNodesXMinus = tMesh->GetNodeSetNodes("x-");

    auto tNumberChildNodes = tNodesXMinus.size();

    Plato::ScalarMultiVector tChildNodeLocations       ("child node locations",        tSpaceDim, tNumberChildNodes);
    Plato::ScalarMultiVector tMappedChildNodeLocations ("mapped child node locations", tSpaceDim, tNumberChildNodes);

    std::vector<std::vector<Plato::Scalar>> tTranslations = { 
     {0.25, 0.5, 1.0},
     {1.0, 0.5, -0.75},
     {0.5, -1.0, 0.25},
     {0.5, -1.0, -0.25}
    };

    auto coords = tMesh->Coordinates();
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumberChildNodes), LAMBDA_EXPRESSION(int nodeOrdinal)
    {
      tChildNodeLocations(0, nodeOrdinal) = coords[nodeOrdinal*tSpaceDim+0];
      tChildNodeLocations(1, nodeOrdinal) = coords[nodeOrdinal*tSpaceDim+1];
      tChildNodeLocations(2, nodeOrdinal) = coords[nodeOrdinal*tSpaceDim+2];

      tMappedChildNodeLocations(0, nodeOrdinal) = coords[nodeOrdinal*tSpaceDim+0] + tTranslations[nodeOrdinal][0];
      tMappedChildNodeLocations(1, nodeOrdinal) = coords[nodeOrdinal*tSpaceDim+1] + tTranslations[nodeOrdinal][1];
      tMappedChildNodeLocations(2, nodeOrdinal) = coords[nodeOrdinal*tSpaceDim+2] + tTranslations[nodeOrdinal][2];
    }, "get coords");

    auto tDomainCellMap = tSpatialModel.Domains.front().cellOrdinals(); // first and only domain

    Plato::OrdinalVector tParentElements("mapped elements", tNumberChildNodes);

    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
      (tSpatialModel.Mesh, tDomainCellMap, tChildNodeLocations, tMappedChildNodeLocations, tParentElements);

    auto tParentElements_Host = Kokkos::create_mirror_view( tParentElements );
    Kokkos::deep_copy( tParentElements_Host, tParentElements );

    std::vector<Plato::OrdinalType> tParentElements_gold = { 2, 5, 4, 3 };

    for(int iParent=0; iParent<int(tParentElements_gold.size()); iParent++){
        TEST_EQUALITY(tParentElements_Host(iParent), tParentElements_gold[iParent]);
    }
}

TEUCHOS_UNIT_TEST(ProjectionTests, ProjectWorksetDisplacements_CompatibleMesh)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh("TET4", tMeshWidth);

    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    
    constexpr int tSpaceDim = ElementType::mNumSpatialDims;
    int tNumCells = tMesh->NumElements();
    constexpr int tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerFace  = ElementType::mNumNodesPerFace;

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
    Plato::SpatialModel tSpatialModel(tMesh, *tInputs);

    // create mesh based displacement from host data
    //
    std::vector<Plato::Scalar> u_host( tSpaceDim*tMesh->NumNodes() );
    Plato::Scalar disp = 0.0, dval = 0.0001;
    for( auto& val : u_host ) val = (disp += dval);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      u_host_view(u_host.data(),u_host.size());
    auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    Plato::ScalarMultiVectorT<Plato::Scalar> tDispWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(u, tDispWS);

    // get side set info
    //
    std::string tSideSetName = "z-";
    auto tChildFaceNodes = tMesh->GetNodeSetNodes(tSideSetName);
    auto tNumChildNodes = tChildFaceNodes.extent(0);
    auto tChildFaceElements = tMesh->GetSideSetElements(tSideSetName);
    auto tNumChildCells = tChildFaceElements.extent(0);
    auto tChildFaceOrdinals = tMesh->GetSideSetFaces(tSideSetName);
    auto tChildFaceLocalNodes = tMesh->GetSideSetLocalNodes(tSideSetName);

    // setting manually, but would need to read translation, or compute initial gap
    Plato::ScalarMultiVector tChildNodeLocations       ("child node locations",        tSpaceDim, tNumChildNodes);
    Plato::ScalarMultiVector tMappedChildNodeLocations ("mapped child node locations", tSpaceDim, tNumChildNodes);

    auto coords = tMesh->Coordinates();
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumChildNodes), LAMBDA_EXPRESSION(int nodeOrdinal)
    {
      auto tNodeOrdinal = tChildFaceNodes(nodeOrdinal);
      tChildNodeLocations(0, nodeOrdinal) = coords[tNodeOrdinal*tSpaceDim+0];
      tChildNodeLocations(1, nodeOrdinal) = coords[tNodeOrdinal*tSpaceDim+1];
      tChildNodeLocations(2, nodeOrdinal) = coords[tNodeOrdinal*tSpaceDim+2];

      tMappedChildNodeLocations(0, nodeOrdinal) = coords[tNodeOrdinal*tSpaceDim+0];
      tMappedChildNodeLocations(1, nodeOrdinal) = coords[tNodeOrdinal*tSpaceDim+1];
      tMappedChildNodeLocations(2, nodeOrdinal) = coords[tNodeOrdinal*tSpaceDim+2] + 1.0;
    }, "get coords");

    // find parent elements
    auto tDomainCellMap = tSpatialModel.Domains.front().cellOrdinals(); // first and only domain
    Plato::OrdinalVector tParentElements("mapped elements", tNumChildNodes);
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
      (tSpatialModel.Mesh, tDomainCellMap, tChildNodeLocations, tMappedChildNodeLocations, tParentElements);

    // project displacements
    Plato::Geometry::GetBasis<ElementType, Plato::Scalar> getBasis(tMesh);
    Plato::InterpolateFromNodal<ElementType, tNumDofsPerNode, /*offset=*/0, tSpaceDim> interpolateFromNodal;

    // to get the state entry ordinal correct, this has to be the full displacement
    // so I have to copy the whole displacement field to only change a few entries
    // it seems like a waste of memory
    Plato::ScalarVector tProjectedDisp("projected displacement", u.extent(0));
    Kokkos::deep_copy(tProjectedDisp, u); // this can just be initialized as 0

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType iChildNode)
    {
        auto tChildNode = tChildFaceNodes(iChildNode);
        auto tParentElement = tParentElements(iChildNode);
        
        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);
        for(Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
        {
            tInPoint(iDim) = tMappedChildNodeLocations(iDim, iChildNode);
        }

        Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tBasis(0.0); // config scalar type
        getBasis(tParentElement, tInPoint, tBasis);

        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tProjectedDisplacement(0.0);
        interpolateFromNodal(tParentElement, tBasis, tDispWS, tProjectedDisplacement);

        for(Plato::OrdinalType iDof=0; iDof<ElementType::mNumDofsPerNode; iDof++)
        {
            tProjectedDisp(tChildNode*ElementType::mNumDofsPerNode + iDof) = tProjectedDisplacement(iDof);
        }
        
    }, "get displacement values of parent face at child node locations");

    // workset projected displacements
    Plato::VectorEntryOrdinal<ElementType::mNumSpatialDims, ElementType::mNumDofsPerNode, ElementType::mNumNodesPerCell> tStateEntryOrdinal(tMesh); /*!< local-to-global ID map for global state */
    Plato::ScalarMultiVectorT<Plato::Scalar> tProjectedDispWS("state workset", tNumChildCells, ElementType::mNumNodesPerFace*ElementType::mNumDofsPerNode);

    // here will I need to workset for Fad type correctly? Or since tProjectedDisp is computed from 
    // tStateWS is it okay?
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumChildCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tCellOrdinal = tChildFaceElements(aCellOrdinal);

        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < ElementType::mNumDofsPerNode; tDofIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerFace; tNodeIndex++)
            {
                auto tLocalNodeOrdinal = tChildFaceLocalNodes(aCellOrdinal*ElementType::mNumNodesPerFace+tNodeIndex);

                Plato::OrdinalType tEntryOrdinal = tStateEntryOrdinal(tCellOrdinal, tLocalNodeOrdinal, tDofIndex);
                Plato::OrdinalType tLocalDof = tNodeIndex * ElementType::mNumDofsPerNode + tDofIndex;
                tProjectedDispWS(aCellOrdinal, tLocalDof) = tProjectedDisp(tEntryOrdinal);
            }
        }
    }, "workset_state_scalar_scalar");

    // TEST workset came out as expected
    //
    std::vector<std::vector<Plato::Scalar>> tProjectedDispWS_gold = { 
    { 0.0004, 0.0005, 0.0006, 0.0010, 0.0011, 0.0012, 0.0022, 0.0023, 0.0024 },
    { 0.0004, 0.0005, 0.0006, 0.0022, 0.0023, 0.0024, 0.0016, 0.0017, 0.0018 }
    };

    auto tProjectedDispWS_Host = Kokkos::create_mirror_view( tProjectedDispWS );
    Kokkos::deep_copy( tProjectedDispWS_Host, tProjectedDispWS );

    for(int iCell=0; iCell<int(tNumChildCells); iCell++){
        for(int iDof=0; iDof<tNumNodesPerFace*tNumDofsPerNode; iDof++){
            TEST_FLOATING_EQUALITY(tProjectedDispWS_Host(iCell,iDof), tProjectedDispWS_gold[iCell][iDof], 1e-12);
      }
    }

    // TEST that displacement difference between projected and child nodes
    // is as expected, i.e. check that worksets line up
    // all projected elements are 1 more than child, meaning they have displacement
    // values 3 higher
    //
    std::vector<std::vector<Plato::Scalar>> tGoldChildMinusProjected = { 
    { 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003 },
    { 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003 }
    };

    auto tDispWS_Host = Kokkos::create_mirror_view( tDispWS );
    Kokkos::deep_copy( tDispWS_Host, tDispWS );

    for(int iCell=0; iCell<int(tNumChildCells); iCell++){
        auto tCellOrdinal = tChildFaceElements(iCell);
        for(int iNode=0; iNode<tNumNodesPerFace; iNode++){
            auto tFullEleWSNodeOrdinal = tChildFaceLocalNodes(iCell*tNumNodesPerFace+iNode);
            for(int iDof=0; iDof<tNumDofsPerNode; iDof++){
                auto tProjectedEleWSDof = iNode*tNumDofsPerNode + iDof;
                auto tFullEleWSDof = tFullEleWSNodeOrdinal*tNumDofsPerNode + iDof;
                auto tDiff = tProjectedDispWS_Host(iCell,tProjectedEleWSDof) - tDispWS_Host(tCellOrdinal,tFullEleWSDof);
                TEST_FLOATING_EQUALITY(tDiff, tGoldChildMinusProjected[iCell][tProjectedEleWSDof], 1e-12);
            }
        }
    }
}

}

