#include "ContactUtils.hpp"
#include "BLAS1.hpp"
#include "alg/CrsMatrixUtils.hpp"

namespace Plato
{

namespace Contact
{

ContactPair parse_contact_pair
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh)
{
    ContactPair tContactPair;

    if (!aParams.isType<std::string>("Side A Child Sideset"))
        ANALYZE_THROWERR("Side A Child Sideset was not provided in contact pair")

    std::string tSideSetA = aParams.get<std::string>("Side A Child Sideset");
    tContactPair.childSideSetA        = tSideSetA;
    tContactPair.childNodesA          = aMesh->GetNodeSetNodes(tSideSetA);
    tContactPair.childElementsA       = aMesh->GetSideSetElements(tSideSetA);
    tContactPair.childFaceLocalNodesA = aMesh->GetSideSetLocalNodes(tSideSetA);

    if (!aParams.isType<std::string>("Side B Child Sideset"))
        ANALYZE_THROWERR("Side B Child Sideset was not provided in contact pair")

    std::string tSideSetB = aParams.get<std::string>("Side B Child Sideset");
    tContactPair.childSideSetB        = tSideSetB;
    tContactPair.childNodesB          = aMesh->GetNodeSetNodes(tSideSetB);
    tContactPair.childElementsB       = aMesh->GetSideSetElements(tSideSetB);
    tContactPair.childFaceLocalNodesB = aMesh->GetSideSetLocalNodes(tSideSetB);
    
    // parse initial gap
    if (!aParams.isType<Teuchos::Array<Plato::Scalar>>("Initial Gap"))
        ANALYZE_THROWERR("Initial Gap vector was not provided in contact pair")

    Plato::OrdinalType tNumDims = aMesh->NumDimensions();
    auto tVector = aParams.get<Teuchos::Array<Plato::Scalar>>("Initial Gap");
    if(tVector.size() != tNumDims)
        ANALYZE_THROWERR("Initial Gap vector provided in contact pair has different dimensions than mesh.")
    
    tContactPair.initialGap = tVector;

    // get spatial domains for finding parent elements
    if (!aParams.isType<std::string>("Side A Block"))
        ANALYZE_THROWERR("Side A Block was not provided in contact pair")

    tContactPair.parentBlockA = aParams.get<std::string>("Side A Block");

    if (!aParams.isType<std::string>("Side B Block"))
        ANALYZE_THROWERR("Side B Block was not provided in contact pair")

    tContactPair.parentBlockB = aParams.get<std::string>("Side B Block");

    return tContactPair;
}

std::vector<ContactPair> parse_contact
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh)
 {
    std::vector<ContactPair> tPairs;
    if (aParams.isSublist("Contact"))
    {
        auto tContactParams = aParams.sublist("Contact");
        if (!tContactParams.isSublist("Pairs"))
        {
            ANALYZE_THROWERR("Parsing 'Contact' parameter list. Required 'Pairs' parameter sublist not found");
        }

        auto tPairsParams = tContactParams.sublist("Pairs");
        for (auto tIndex = tPairsParams.begin(); tIndex != tPairsParams.end(); ++tIndex)
        {
            const auto &tEntry  = tPairsParams.entry(tIndex);
            const auto &tMyName = tPairsParams.name(tIndex);

            if (!tEntry.isList())
            {
                ANALYZE_THROWERR("Parameter in 'Domains' parameter sublist within 'Spatial Model' parameter list not valid.  Expect lists only.");
            }

            Teuchos::ParameterList &tPairParams = tPairsParams.sublist(tMyName);
            tPairs.push_back(parse_contact_pair(tPairParams, aMesh));
        }
    }
    return tPairs;
 }

Plato::SpatialDomain get_domain
(const std::string                       & aDomainName,
 const std::vector<Plato::SpatialDomain> & aDomains)
{
    for(auto& tDomain : aDomains)
    {
        auto tName = tDomain.getElementBlockName();
        if( tName == aDomainName )
            return tDomain;
    }
    std::string tMsg = "Block with name " + aDomainName + " provided in get_domain does not correspond to an element block name in spatial model";
    ANALYZE_THROWERR(tMsg)
}

Plato::ScalarMultiVector compute_node_locations
(Plato::Mesh                                             aMesh,
 const Plato::OrdinalVectorT<const Plato::OrdinalType> & aNodes)
{
    Plato::OrdinalType tNumNodes = aNodes.size();
    Plato::OrdinalType tSpaceDim = aMesh->NumDimensions();
    Plato::ScalarMultiVector tLocations("node locations", tSpaceDim, tNumNodes);

    auto tCoords = aMesh->Coordinates();
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(int nodeOrdinal)
    {
        auto tNodeOrdinal = aNodes(nodeOrdinal);
        for (Plato::OrdinalType iDim = 0; iDim < tSpaceDim; iDim++)
            tLocations(iDim, nodeOrdinal) = tCoords(tNodeOrdinal*tSpaceDim+iDim);
    }, "get coords");

    return tLocations;
}

Plato::ScalarMultiVector map_node_locations
(const Plato::ScalarMultiVector      & aLocations,
 const Teuchos::Array<Plato::Scalar> & aTranslation,
 Plato::Scalar                         aScale)
{
    static constexpr int tSpaceDim = 3;
    if (aTranslation.size() != tSpaceDim || aLocations.extent(0) != tSpaceDim)
        ANALYZE_THROWERR("In ContactUtils map_node_locations, an incorrect dimension is given. Only 3 dimensions are supported.")
    
    Plato::Scalar tTranslationX = aScale * aTranslation[0];
    Plato::Scalar tTranslationY = aScale * aTranslation[1];
    Plato::Scalar tTranslationZ = aScale * aTranslation[2];

    Plato::OrdinalType tNumNodes = aLocations.extent(1);
    Plato::ScalarMultiVector tMappedLocations("mapped node locations", tSpaceDim, tNumNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(int nodeOrdinal)
    {
        tMappedLocations(0, nodeOrdinal) = aLocations(0, nodeOrdinal) + tTranslationX;
        tMappedLocations(1, nodeOrdinal) = aLocations(1, nodeOrdinal) + tTranslationY;
        tMappedLocations(2, nodeOrdinal) = aLocations(2, nodeOrdinal) + tTranslationZ;
    }, "map coords");

    return tMappedLocations;
}

Plato::OrdinalVector global_local_child_node_ord_map
(const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildNodes,
       Plato::OrdinalType                                aNumMeshNodes)
{
    Plato::OrdinalVector tMap("map from global child node ordinal to local child node array ordinal", aNumMeshNodes);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), tMap);  

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,aChildNodes.size()), KOKKOS_LAMBDA(int nodeOrdinal)
    {
        Plato::OrdinalType tOrdinal = aChildNodes(nodeOrdinal);
        tMap(tOrdinal) = nodeOrdinal;
    }, "");

    return tMap;
}

Plato::OrdinalVector convert_to_elementwise_map
(const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildElements,
 const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildFaceLocalNodes,
 const Plato::OrdinalVector                            & aMap,
       Plato::Mesh                                       aMesh,
       Plato::OrdinalType                                aNumNodesPerFace)
{
    auto tNumChildElements = aChildElements.size();
    auto tNumNodesPerElement = aMesh->NumNodesPerElement();
    auto tConnectivity = aMesh->Connectivity();

    Plato::OrdinalVector tElementWiseMap("element-wise storage of map",aChildElements.size()*aNumNodesPerFace);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumChildElements), KOKKOS_LAMBDA(const Plato::OrdinalType & cellOrdinal)
    {
        auto tCellOrdinal = aChildElements(cellOrdinal);

        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < aNumNodesPerFace; tNodeIndex++)
        {
            auto tLocalNodeOrdinal = aChildFaceLocalNodes(cellOrdinal*aNumNodesPerFace+tNodeIndex);
            auto tGlobalNodeOrdinal = tConnectivity(tCellOrdinal*tNumNodesPerElement + tLocalNodeOrdinal);
            tElementWiseMap(cellOrdinal*aNumNodesPerFace + tNodeIndex) = aMap(tGlobalNodeOrdinal);
        }
    }, "");

    return tElementWiseMap;
}

Plato::OrdinalType count_total_child_nodes(const std::vector<ContactPair> & aPairs)
{
    Plato::OrdinalType tNum(0);
    for (auto tPair : aPairs)
        tNum += tPair.childNodesA.size() + tPair.childNodesB.size();
    
    return tNum;
}

void check_for_repeated_child_nodes
(const Plato::OrdinalVector & aChildNodes,
       Plato::Mesh            aMesh)
{
    auto tNumTotalNodes = aMesh->NumNodes();
    Plato::OrdinalVector tCheckChildNodes("", tNumTotalNodes);

    auto tNumChildNodes = aChildNodes.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
    {
        auto tChildNode = aChildNodes(nodeOrdinal);
        Kokkos::atomic_increment(&tCheckChildNodes(tChildNode));
    }, "");

    Plato::OrdinalType tNumRecordedChild(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumTotalNodes),
    KOKKOS_LAMBDA(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
    {
        Kokkos::atomic_add(&aUpdate, tCheckChildNodes(aOrdinal));
    }, tNumRecordedChild);
    if ( tNumRecordedChild != tNumChildNodes )
    {
        ANALYZE_THROWERR("REPEATED CHILD NODE IN CONTACT SURFACE PAIRS")
    }
}

Teuchos::RCP<Plato::CrsMatrixType> add_contact_graph_to_matrix
(Teuchos::RCP<Plato::CrsMatrixType>   aMatrix,
 Plato::Mesh                          aMesh,
 const Plato::OrdinalVector         & aChildNodes,
 const Plato::OrdinalVector         & aParentElements)
{
    if (!aMatrix->isBlockMatrix())
        ANALYZE_THROWERR("NON-BLOCK MATRIX WAS PASSED TO add_contact_graph_to_matrix FUNCTION")

    auto tOffsetMap = aMatrix->rowMap();
    auto tNodeOrds  = aMatrix->columnIndices();

    auto tNumTotalNodes = aMesh->NumNodes();
    auto tNumChildNodes = aChildNodes.size();

    // find and store number of entries in node node graph for just child nodes
    Plato::OrdinalVector tChildOffsetMap("child node offset map", tNumChildNodes+1);
    Plato::OrdinalType tNumChildConnectedNodes(0);

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
    }, tNumChildConnectedNodes);

    // mark child nodes
    Plato::OrdinalVector tMarkedNodes("marking child nodes", tNumTotalNodes);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), tMarkedNodes);  

    auto tNumNodesPerCell  = aMesh->NumNodesPerElement();
    auto tConnectivity = aMesh->Connectivity();

    Plato::OrdinalType tNumOrdinals = tNumChildConnectedNodes*tNumNodesPerCell;
    Plato::OrdinalVector tFatGraph_ordinals("largest number of possible nodes in graph", tNumOrdinals);
    Plato::OrdinalVector tNumConnectedNodes("number of nodes connected by contact", tNumChildNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType iChildNode)
    {
        Plato::OrdinalType tNumUnique(0);

        auto tChildNode = aChildNodes(iChildNode);
        tMarkedNodes(tChildNode) = iChildNode;
                
        Plato::OrdinalType tFrom = tOffsetMap(tChildNode);
        Plato::OrdinalType tTo   = tOffsetMap(tChildNode + 1);

        auto tFatGraphOffset = tChildOffsetMap(iChildNode)*tNumNodesPerCell;

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
                for(Plato::OrdinalType tElemLocalNodeOrd=0; tElemLocalNodeOrd<tNumNodesPerCell; tElemLocalNodeOrd++)
                {
                    auto tNodeOrd = tConnectivity(tParentElement*tNumNodesPerCell + tElemLocalNodeOrd);

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

    Plato::OrdinalVector tFullOffsetMap("offset map accounting for contact", tNumTotalNodes+1);

    Plato::OrdinalType tNumNodeNodeEntries(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumTotalNodes),
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
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumTotalNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
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
        auto tChildMark = tMarkedNodes(aNodeOrdinal);
        if (tChildMark >= 0)
        {
            auto tNewConnected = tNumConnectedNodes(tChildMark);
            auto tStart = tNewFrom + tOldNum;
            auto tEnd   = tStart + tNewConnected;

            auto tFatGraphOffset = tChildOffsetMap(tChildMark)*tNumNodesPerCell;
            for( Plato::OrdinalType tIndex=tStart; tIndex<tEnd; tIndex++ )
            {
                tFullNodeOrds(tIndex) = tFatGraph_ordinals(tFatGraphOffset++);
            }
        }
    }, "node ordinals accounting for contact");

    Plato::sort_matrix_column_ordinals(tFullOffsetMap, tFullNodeOrds);

    // create new matrix
    auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aMatrix->numColsPerBlock();
    auto numRows = tFullOffsetMap.size() - 1;
    auto nnz = tFullNodeOrds.size();
    Plato::OrdinalType numBlockDofs = tNumRowsPerBlock*tNumColsPerBlock;
    typename Plato::CrsMatrixType::ScalarVectorT entries("matrix entries", nnz*numBlockDofs);
    auto retMatrix = Teuchos::rcp(
     new Plato::CrsMatrixType( tFullOffsetMap, tFullNodeOrds, entries,
                     numRows*tNumRowsPerBlock, numRows*tNumColsPerBlock,
                     tNumRowsPerBlock, tNumColsPerBlock )
    );
    return retMatrix;
}

}

}
