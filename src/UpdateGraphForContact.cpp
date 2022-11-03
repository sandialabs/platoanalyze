#include "UpdateGraphForContact.hpp"
#include "BLAS1.hpp"
#include "alg/CrsMatrixUtils.hpp"

namespace Plato
{

namespace Contact
{
UpdateGraphForContact::UpdateGraphForContact
(Plato::Mesh                  aMesh,
 const Plato::OrdinalVector & aChildNodes,
 const Plato::OrdinalVector & aParentElements) : 
 mChildNodes(aChildNodes),
 mParentElements(aParentElements),
 mConnectivity(aMesh->Connectivity()),
 mNumTotalNodes(aMesh->NumNodes()),
 mNumNodesPerElement(aMesh->NumNodesPerElement()),
 mChildOffsetMap("child node offset map", aChildNodes.size() + 1),
 mMarkedChildNodes("marking child nodes", aMesh->NumNodes()),
 mNumConnectedNodes("number of nodes connected by contact", aChildNodes.size()),
 mAllGraphOrdinals("largest number of possible nodes in graph, has repeated values", 0),
 mFullOffsetMap("offset map accounting for contact", aMesh->NumNodes() + 1),
 mFullNodeOrds("node-node ordinals accounting for contact", 0)
{
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), mMarkedChildNodes);  
}

Teuchos::RCP<Plato::CrsMatrixType> 
UpdateGraphForContact::operator() (Teuchos::RCP<Plato::CrsMatrixType> aMatrix)
{
    if (!aMatrix->isBlockMatrix())
        ANALYZE_THROWERR("UpdateGraphForContact functor expected input matrix to be a block matrix.")

    auto tOffsetMap = aMatrix->rowMap();
    auto tNodeOrds  = aMatrix->columnIndices();

    auto tNumChildConnectedNodes = this->extractChildNodeOffsets(tOffsetMap);
    Plato::OrdinalType tNumOrdinals = tNumChildConnectedNodes*mNumNodesPerElement;
    Kokkos::resize(mAllGraphOrdinals, tNumOrdinals);

    this->storeUniqueParentNodeContributions(tOffsetMap, tNodeOrds);

    auto tNumNodeNodeEntries = this->updateOffsetMap(tOffsetMap);
    Kokkos::resize(mFullNodeOrds, tNumNodeNodeEntries);

    updateNodeOrds(tOffsetMap, tNodeOrds);

    Plato::sort_matrix_column_ordinals(mFullOffsetMap, mFullNodeOrds);

    // create new matrix
    auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aMatrix->numColsPerBlock();
    auto numRows = mFullOffsetMap.size() - 1;
    auto nnz = mFullNodeOrds.size();
    Plato::OrdinalType numBlockDofs = tNumRowsPerBlock*tNumColsPerBlock;
    typename Plato::CrsMatrixType::ScalarVectorT entries("matrix entries", nnz*numBlockDofs);
    auto retMatrix = Teuchos::rcp(
     new Plato::CrsMatrixType( mFullOffsetMap, mFullNodeOrds, entries,
                     numRows*tNumRowsPerBlock, numRows*tNumColsPerBlock,
                     tNumRowsPerBlock, tNumColsPerBlock )
    );
    return retMatrix;
}

Plato::OrdinalType 
UpdateGraphForContact::extractChildNodeOffsets(const Plato::OrdinalVector & aOffsetMap)
{
    auto tNumChildNodes = mChildNodes.size();

    auto& tChildNodes = mChildNodes;
    auto& tChildOffsetMap = mChildOffsetMap;

    Plato::OrdinalType tTotalConnectedNodes(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumChildNodes),
    KOKKOS_LAMBDA (const Plato::OrdinalType& aOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        auto tChildNode = tChildNodes(aOrdinal);
        const auto tNumConnected = aOffsetMap(tChildNode+1) - aOffsetMap(tChildNode);

        aUpdate += tNumConnected;
        if( tIsFinal )
        {
          tChildOffsetMap(aOrdinal+1) = aUpdate;
        }
    }, tTotalConnectedNodes);

    return tTotalConnectedNodes;
}

void 
UpdateGraphForContact::storeUniqueParentNodeContributions
(const Plato::OrdinalVector & aOffsetMap, 
 const Plato::OrdinalVector & aNodeOrds)
{
    auto tNumChildNodes = mChildNodes.size();
    
    auto tNumNodesPerElement = mNumNodesPerElement;
    auto& tChildNodes = mChildNodes;
    auto& tParentElements = mParentElements;
    auto& tMarkedChildNodes = mMarkedChildNodes;
    auto& tChildOffsetMap = mChildOffsetMap;
    auto& tConnectivity = mConnectivity;
    auto& tAllGraphOrdinals = mAllGraphOrdinals;
    auto& tNumConnectedNodes = mNumConnectedNodes;

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType iChildNode)
    {
        Plato::OrdinalType tNumUnique(0);

        auto tChildNode = tChildNodes(iChildNode);
        tMarkedChildNodes(tChildNode) = iChildNode;
                
        Plato::OrdinalType tFrom = aOffsetMap(tChildNode);
        Plato::OrdinalType tTo   = aOffsetMap(tChildNode + 1);

        auto tFatGraphOffset = tChildOffsetMap(iChildNode)*tNumNodesPerElement;

        for(Plato::OrdinalType iOrd=tFrom; iOrd<tTo; iOrd++)
        {
            auto tGraphNode = aNodeOrds(iOrd);
            
            // check if node in graph is a child node
            Plato::OrdinalType tOutput = -1;
            for(Plato::OrdinalType iChild=0; iChild<tNumChildNodes; iChild++)
            {
                if (tChildNodes(iChild) == tGraphNode)
                {
                    tOutput = iChild;
                    break;
                }
            }

            if (tOutput >= 0)
            {
                auto tParentElement = tParentElements(tOutput);
                for(Plato::OrdinalType tElemLocalNodeOrd=0; tElemLocalNodeOrd<tNumNodesPerElement; tElemLocalNodeOrd++)
                {
                    auto tNodeOrd = tConnectivity(tParentElement*tNumNodesPerElement + tElemLocalNodeOrd);

                    // get unique parent nodes
                    bool isUnique = true;
                    for( Plato::OrdinalType tIndex=0; tIndex<tNumUnique; tIndex++ )
                    {
                        if( tAllGraphOrdinals(tFatGraphOffset+tIndex) == tNodeOrd )
                            isUnique = false;
                    }
                    if(isUnique)
                    {
                        tAllGraphOrdinals(tFatGraphOffset+tNumUnique) = tNodeOrd;
                        tNumUnique++;
                    }
                }
            }
        }
        tNumConnectedNodes(iChildNode) = tNumUnique;
    });
}

Plato::OrdinalType 
UpdateGraphForContact::updateOffsetMap(const Plato::OrdinalVector & aOffsetMap)
{
    auto tNumTotalNodes = mFullOffsetMap.size() - 1;
    auto& tFullOffsetMap = mFullOffsetMap;
    auto& tMarkedChildNodes = mMarkedChildNodes;
    auto& tNumConnectedNodes = mNumConnectedNodes;

    Plato::OrdinalType tNumOffsets(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumTotalNodes),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        auto tChildMark = tMarkedChildNodes(iOrdinal);
        
        auto tOriginalNum = aOffsetMap(iOrdinal+1) - aOffsetMap(iOrdinal);
        auto tContactNum = tNumConnectedNodes(tChildMark);

        const auto tVal = (tChildMark < 0) ? tOriginalNum : tOriginalNum + tContactNum;
        aUpdate += tVal;
        if( tIsFinal )
        {
          tFullOffsetMap(iOrdinal+1) = aUpdate;
        }
    }, tNumOffsets);
    
    return tNumOffsets;
}

void 
UpdateGraphForContact::updateNodeOrds
(const Plato::OrdinalVector & aOffsetMap, 
 const Plato::OrdinalVector & aNodeOrds)
{
    auto tNumTotalNodes = mFullOffsetMap.size() - 1;
    auto tNumNodesPerElement = mNumNodesPerElement;
    auto& tFullOffsetMap = mFullOffsetMap;
    auto& tFullNodeOrds = mFullNodeOrds;
    auto& tMarkedChildNodes = mMarkedChildNodes;
    auto& tNumConnectedNodes = mNumConnectedNodes;
    auto& tChildOffsetMap = mChildOffsetMap;
    auto& tAllGraphOrdinals = mAllGraphOrdinals;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumTotalNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
    {
        auto tNewFrom = tFullOffsetMap(aNodeOrdinal);

        // fill in old entries
        auto tOldFrom = aOffsetMap(aNodeOrdinal);
        auto tOldNum  = aOffsetMap(aNodeOrdinal+1) - tOldFrom;
        for( Plato::OrdinalType tIndex=0; tIndex<tOldNum; tIndex++ )
        {
            tFullNodeOrds(tNewFrom+tIndex) = aNodeOrds(tOldFrom+tIndex);
        }

        // fill in new entries
        auto tChildMark = tMarkedChildNodes(aNodeOrdinal);
        if (tChildMark >= 0)
        {
            auto tNewConnected = tNumConnectedNodes(tChildMark);
            auto tStart = tNewFrom + tOldNum;
            auto tEnd   = tStart + tNewConnected;

            auto tFatGraphOffset = tChildOffsetMap(tChildMark)*tNumNodesPerElement;
            for( Plato::OrdinalType tIndex=tStart; tIndex<tEnd; tIndex++ )
            {
                tFullNodeOrds(tIndex) = tAllGraphOrdinals(tFatGraphOffset++);
            }
        }
    }, "node ordinals accounting for contact");
}

}

}
