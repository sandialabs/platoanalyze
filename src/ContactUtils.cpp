#include "ContactUtils.hpp"
#include "ContactPair.hpp"
#include "BLAS1.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Contact
{

ContactPair parse_contact_pair
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh)
{
    ContactPair tContactPair;

    if (aParams.isSublist("A Surface"))
    {
        auto tSurfaceParams = aParams.sublist("A Surface");
        tContactPair.surfaceA.initialize(tSurfaceParams, aMesh);
    }
    else
        ANALYZE_THROWERR("Parsing 'Contact' parameter list 'Pairs' sublist. Required 'A Surface' parameter sublist not found");

    if (aParams.isSublist("B Surface"))
    {
        auto tSurfaceParams = aParams.sublist("B Surface");
        tContactPair.surfaceB.initialize(tSurfaceParams, aMesh);
    }
    else
        ANALYZE_THROWERR("Parsing 'Contact' parameter list 'Pairs' sublist. Required 'B Surface' parameter sublist not found");

    if (!aParams.isType<Teuchos::Array<Plato::Scalar>>("Initial Gap"))
        ANALYZE_THROWERR("Parsing 'Contact' parameter list 'Pairs' sublist. Required 'Initial Gap' parameter not found")

    Plato::OrdinalType tNumDims = aMesh->NumDimensions();
    auto tVector = aParams.get<Teuchos::Array<Plato::Scalar>>("Initial Gap");
    if(tVector.size() != tNumDims)
        ANALYZE_THROWERR("Initial Gap vector provided in contact pair has different dimensions than mesh.")
    tContactPair.initialGap = tVector;

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
 const Teuchos::Array<Plato::Scalar> & aTranslation)
{
    static constexpr int tSpaceDim = 3;
    if (aTranslation.size() != tSpaceDim || aLocations.extent(0) != tSpaceDim)
        ANALYZE_THROWERR("In ContactUtils map_node_locations, an incorrect dimension is given. Only 3 dimensions are supported.")
    
    Plato::Scalar tTranslationX = aTranslation[0];
    Plato::Scalar tTranslationY = aTranslation[1];
    Plato::Scalar tTranslationZ = aTranslation[2];

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

Teuchos::Array<Plato::Scalar> 
scale_initial_gap
(const Teuchos::Array<Plato::Scalar> aGap,
 Plato::Scalar                       aScale)
{
    Teuchos::Array<Plato::Scalar> tScaledGap = aGap;

    for (Plato::OrdinalType iDim = 0; iDim < aGap.size(); iDim++)
    {
        tScaledGap[iDim] *= aScale;
    }

    return tScaledGap;
}

Plato::OrdinalType count_total_child_nodes(const std::vector<ContactPair> & aPairs)
{
    Plato::OrdinalType tNum(0);
    for (auto tPair : aPairs)
        tNum += tPair.surfaceA.childNodes().size() + tPair.surfaceB.childNodes().size();
    
    return tNum;
}

void populate_full_contact_arrays
(const std::vector<ContactPair> & aPairs,
       Plato::OrdinalVector     & aChildNodes,
       Plato::OrdinalVector     & aParentElements)
{
    Plato::OrdinalType tOffset(0);
    for (auto tPair : aPairs)
    {
        auto tChildNodes = tPair.surfaceA.childNodes();
        auto tParentElements = tPair.surfaceA.parentElements();
        Plato::OrdinalType tNumNodes = tChildNodes.size();
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
        {
            aChildNodes(tOffset + nodeOrdinal) = tChildNodes(nodeOrdinal);
            aParentElements(tOffset + nodeOrdinal) = tParentElements(nodeOrdinal);
        }, "store child nodes and parent elements");
        tOffset += tNumNodes;

        tChildNodes = tPair.surfaceB.childNodes();
        tParentElements = tPair.surfaceB.parentElements();
        tNumNodes = tChildNodes.size();
        auto tScaledGap = scale_initial_gap(tPair.initialGap, -1.0);
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
        {
            aChildNodes(tOffset + nodeOrdinal) = tChildNodes(nodeOrdinal);
            aParentElements(tOffset + nodeOrdinal) = tParentElements(nodeOrdinal);
        }, "store child nodes and parent elements");
        tOffset += tNumNodes;
    }
}

void check_for_repeated_child_nodes
(const Plato::OrdinalVector & aChildNodes,
       Plato::OrdinalType     aNumMeshNodes)
{
    Plato::OrdinalVector tCheckChildNodes("", aNumMeshNodes);

    auto tNumChildNodes = aChildNodes.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
    {
        auto tChildNode = aChildNodes(nodeOrdinal);
        Kokkos::atomic_increment(&tCheckChildNodes(tChildNode));
    }, "");

    Plato::OrdinalType tNumRepeatedChild(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, aNumMeshNodes),
    KOKKOS_LAMBDA(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
    {
        if ( tCheckChildNodes(aOrdinal) > 1 ) 
        {
            Kokkos::atomic_increment(&aUpdate);
        }
    }, tNumRepeatedChild);
    if ( tNumRepeatedChild != 0 )
    {
        ANALYZE_THROWERR("REPEATED CHILD NODE IN CONTACT SURFACE PAIRS")
    }
}

}

}
