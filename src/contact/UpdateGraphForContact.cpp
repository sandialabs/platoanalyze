#include "contact/UpdateGraphForContact.hpp"
#include "BLAS1.hpp"
#include "alg/CrsMatrixUtils.hpp"

#include <KokkosSparse_SortCrs.hpp>

namespace Plato
{

namespace Contact
{
UpdateGraphForContact::UpdateGraphForContact(Plato::Mesh aMesh) :
 mConnectivity(aMesh->Connectivity()),
 mNumTotalNodes(aMesh->NumNodes()),
 mNumNodesPerElement(aMesh->NumNodesPerElement()),
 mMarkedChildNodes("marking child nodes", aMesh->NumNodes()),
 mOffsetMap("offset map before contact", 0),
 mNodeOrds("node-node ordinals before contact", 0),
 mFullOffsetMap("offset map accounting for contact", aMesh->NumNodes() + 1),
 mFullNodeOrds("node-node ordinals accounting for contact", 0),
 mChildOffsetMap("child node offset map", 0),
 mNumConnectedNodes("number of nodes connected by contact", 0),
 mAllGraphOrdinals("largest number of possible nodes in graph, has repeated values", 0)
{
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), mMarkedChildNodes);  
    aMesh->NodeNodeGraph(mOffsetMap, mNodeOrds);
}

void 
UpdateGraphForContact::createNodeNodeGraph
(const Plato::OrdinalVector & aChildNodes,
 const Plato::OrdinalVector & aParentElements)
{
    Kokkos::resize(mChildOffsetMap, aChildNodes.size() + 1);
    Kokkos::resize(mNumConnectedNodes, aChildNodes.size());

    auto tNumChildConnectedNodes = this->extractChildNodeOffsets(aChildNodes);
    Plato::OrdinalType tNumOrdinals = tNumChildConnectedNodes*mNumNodesPerElement;
    Kokkos::resize(mAllGraphOrdinals, tNumOrdinals);

    this->storeUniqueParentNodeContributions(aChildNodes, aParentElements);

    auto tNumNodeNodeEntries = this->updateOffsetMap();
    Kokkos::resize(mFullNodeOrds, tNumNodeNodeEntries);

    this->updateNodeOrds();

    KokkosSparse::sort_crs_graph<Plato::ExecSpace>(mFullOffsetMap, mFullNodeOrds);
}

void
UpdateGraphForContact::NodeNodeGraph
(Plato::OrdinalVector & aOffsetMap,
 Plato::OrdinalVector & aNodeOrds) const
 {
    aOffsetMap = mFullOffsetMap;
    aNodeOrds = mFullNodeOrds;
 }

void
UpdateGraphForContact::NodeNodeGraphTranspose
(Plato::OrdinalVector & aOffsetMap,
 Plato::OrdinalVector & aNodeOrds) const
 {
    this->countNonzerosForTranspose(aOffsetMap);
    auto tNumEntries = this->constructTransposeOffsetMap(aOffsetMap);

    Kokkos::resize(aNodeOrds, tNumEntries);
    this->constructTransposeNodeOrds(aOffsetMap, aNodeOrds);

    KokkosSparse::sort_crs_graph<Plato::ExecSpace>(aOffsetMap, aNodeOrds);
 }

Plato::OrdinalType 
UpdateGraphForContact::extractChildNodeOffsets(const Plato::OrdinalVector & aChildNodes)
{
    auto tNumChildNodes = aChildNodes.size();

    auto tOffsetMap = mOffsetMap;
    auto tChildOffsetMap = mChildOffsetMap;

    Plato::OrdinalType tTotalConnectedNodes(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumChildNodes),
    KOKKOS_LAMBDA (const Plato::OrdinalType& aOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        auto tChildNode = aChildNodes(aOrdinal);
        const auto tNumConnected = tOffsetMap(tChildNode+1) - tOffsetMap(tChildNode);

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
(const Plato::OrdinalVector & aChildNodes, 
 const Plato::OrdinalVector & aParentElements)
{
    auto tNumChildNodes = aChildNodes.size();
    
    auto tNumNodesPerElement = mNumNodesPerElement;
    auto tMarkedChildNodes = mMarkedChildNodes;
    auto tOffsetMap = mOffsetMap;
    auto tNodeOrds = mNodeOrds;
    auto tChildOffsetMap = mChildOffsetMap;
    auto tConnectivity = mConnectivity;
    auto tAllGraphOrdinals = mAllGraphOrdinals;
    auto tNumConnectedNodes = mNumConnectedNodes;

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType iChildNode)
    {
        Plato::OrdinalType tNumUnique(0);

        auto tChildNode = aChildNodes(iChildNode);
        tMarkedChildNodes(tChildNode) = iChildNode;
                
        Plato::OrdinalType tFrom = tOffsetMap(tChildNode);
        Plato::OrdinalType tTo   = tOffsetMap(tChildNode + 1);

        auto tFatGraphOffset = tChildOffsetMap(iChildNode)*tNumNodesPerElement;

        for(Plato::OrdinalType iOrd=tFrom; iOrd<tTo; iOrd++)
        {
            auto tGraphNode = tNodeOrds(iOrd);
            
            // check if node in graph is a child node
            Plato::OrdinalType tOutput = -1;
            for(Plato::OrdinalType iChild=0; iChild<tNumChildNodes; iChild++)
            {
                if (aChildNodes(iChild) == tGraphNode)
                {
                    tOutput = iChild;
                    break;
                }
            }

            if (tOutput >= 0)
            {
                auto tParentElement = aParentElements(tOutput);
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
UpdateGraphForContact::updateOffsetMap()
{
    auto tNumTotalNodes = mFullOffsetMap.size() - 1;
    auto tOffsetMap = mOffsetMap;
    auto tFullOffsetMap = mFullOffsetMap;
    auto tMarkedChildNodes = mMarkedChildNodes;
    auto tNumConnectedNodes = mNumConnectedNodes;

    Plato::OrdinalType tNumOffsets(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumTotalNodes),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        auto tChildMark = tMarkedChildNodes(iOrdinal);
        
        auto tOriginalNum = tOffsetMap(iOrdinal+1) - tOffsetMap(iOrdinal);
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
UpdateGraphForContact::updateNodeOrds()
{
    auto tNumTotalNodes = mFullOffsetMap.size() - 1;
    auto tNumNodesPerElement = mNumNodesPerElement;
    auto tOffsetMap = mOffsetMap;
    auto tNodeOrds = mNodeOrds;
    auto tFullOffsetMap = mFullOffsetMap;
    auto tFullNodeOrds = mFullNodeOrds;
    auto tMarkedChildNodes = mMarkedChildNodes;
    auto tNumConnectedNodes = mNumConnectedNodes;
    auto tChildOffsetMap = mChildOffsetMap;
    auto tAllGraphOrdinals = mAllGraphOrdinals;

    Kokkos::parallel_for("node ordinals accounting for contact",
                         Kokkos::RangePolicy<>(0, tNumTotalNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
    {
        auto tNewFrom = tFullOffsetMap(aNodeOrdinal);

        // fill in old entries
        auto tOldFrom = tOffsetMap(aNodeOrdinal);
        auto tOldNum  = tOffsetMap(aNodeOrdinal+1) - tOldFrom;
        for( Plato::OrdinalType tIndex=0; tIndex<tOldNum; tIndex++ )
        {
            tFullNodeOrds(tNewFrom+tIndex) = tNodeOrds(tOldFrom+tIndex);
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
    });
}

void 
UpdateGraphForContact::countNonzerosForTranspose
(Plato::OrdinalVector & aOffsetMap) const
{
    auto tFullOffsetMap = mFullOffsetMap;
    auto tFullNodeOrds = mFullNodeOrds;

    Kokkos::resize(aOffsetMap, mFullOffsetMap.size());
    Plato::OrdinalType tNumTotalNodes = aOffsetMap.size() - 1;
    Kokkos::parallel_for("nonzeros", Kokkos::RangePolicy<OrdinalType>(0, tNumTotalNodes), KOKKOS_LAMBDA(OrdinalType iNodeOrdinal)
    {
        auto tFrom = tFullOffsetMap(iNodeOrdinal);
        auto tTo = tFullOffsetMap(iNodeOrdinal + 1);
        for (auto tEntryIndex = tFrom; tEntryIndex < tTo; tEntryIndex++)
        {
            auto iColumnIndex = tFullNodeOrds(tEntryIndex);
            Kokkos::atomic_increment(&aOffsetMap(iColumnIndex));
        }
    });
}

Plato::OrdinalType 
UpdateGraphForContact::constructTransposeOffsetMap
(Plato::OrdinalVector & aOffsetMap) const
{
    Plato::OrdinalType tNumTotalNodes = aOffsetMap.size() - 1;
    Plato::OrdinalType tNumEntries(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalType>(0,tNumTotalNodes+1),
    KOKKOS_LAMBDA (const OrdinalType& iOrdinal, OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const OrdinalType tVal = aOffsetMap(iOrdinal);
        if( tIsFinal )
        {
            aOffsetMap(iOrdinal) = aUpdate;
        }
        aUpdate += tVal;
    }, tNumEntries);

    return tNumEntries;
}

void
UpdateGraphForContact::constructTransposeNodeOrds
(const Plato::OrdinalVector & aOffsetMap,
       Plato::OrdinalVector & aNodeOrds) const
{
    auto tFullOffsetMap = mFullOffsetMap;
    auto tFullNodeOrds = mFullNodeOrds;

    Plato::OrdinalType tNumTotalNodes = aOffsetMap.size() - 1;
    Plato::OrdinalVector tOffsetT("offsets", tNumTotalNodes);
    Kokkos::parallel_for("node ords", Kokkos::RangePolicy<OrdinalType>(0, tNumTotalNodes), KOKKOS_LAMBDA(OrdinalType iNodeOrdinal)
    {
        auto tFrom = tFullOffsetMap(iNodeOrdinal);
        auto tTo = tFullOffsetMap(iNodeOrdinal + 1);
        for (auto iEntryIndex = tFrom; iEntryIndex < tTo; iEntryIndex++)
        {
            auto iRowIndexT = tFullNodeOrds(iEntryIndex);
            auto tMyOffset = Kokkos::atomic_fetch_add(&tOffsetT(iRowIndexT), 1);
            auto iEntryIndexT = aOffsetMap(iRowIndexT) + tMyOffset;
            aNodeOrds(iEntryIndexT) = iNodeOrdinal;
        }
    });
}

}

}
