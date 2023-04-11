#include "contact/ContactUtils.hpp"
#include "contact/ContactPair.hpp"
#include "BLAS1.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Contact
{

std::vector<ContactPair> parse_contact
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh)
 {
    std::vector<ContactPair> tPairs;
    if (!aParams.isSublist("Pairs"))
    {
        ANALYZE_THROWERR("Parsing 'Contact' parameter list. Required 'Pairs' parameter sublist not found");
    }

    auto tPairsParams = aParams.sublist("Pairs");
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
    return tPairs;
 }

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

    if (aParams.isType<Plato::Scalar>("Search Tolerance"))
        tContactPair.searchTolerance = aParams.get<Plato::Scalar>("Search Tolerance");
    else
        tContactPair.searchTolerance = -1.0;

    parse_contact_penalty(aParams, tContactPair);

    return tContactPair;
}

void parse_contact_penalty
(const Teuchos::ParameterList & aParams,
       ContactPair            & aContactPair)
{
    if (!aParams.isType<std::string>("Penalty Type"))
        ANALYZE_THROWERR("Parsing 'Contact' parameter list 'Pairs' sublist. Required 'Penalty Type' parameter not found")
    aContactPair.penaltyType = aParams.get<std::string>("Penalty Type");

    if (aParams.isType<Plato::Scalar>("Penalty Value"))
    {
        Plato::Scalar tValue = aParams.get<Plato::Scalar>("Penalty Value");
        Teuchos::Array<Plato::Scalar> tValueArray(1, tValue);
        aContactPair.penaltyValue = tValueArray;
    }
    else if (aParams.isType<Teuchos::Array<Plato::Scalar>>("Penalty Value"))
    {
        aContactPair.penaltyValue = aParams.get<Teuchos::Array<Plato::Scalar>>("Penalty Value");
    }
    else
        ANALYZE_THROWERR("Parsing 'Contact' parameter list 'Pairs' sublist. Required 'Penalty Value' parameter not found")
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
    Kokkos::parallel_for("get coords", Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(int nodeOrdinal)
    {
        auto tNodeOrdinal = aNodes(nodeOrdinal);
        for (Plato::OrdinalType iDim = 0; iDim < tSpaceDim; iDim++)
            tLocations(iDim, nodeOrdinal) = tCoords(tNodeOrdinal*tSpaceDim+iDim);
    });

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

    Kokkos::parallel_for("map coords", Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(int nodeOrdinal)
    {
        tMappedLocations(0, nodeOrdinal) = aLocations(0, nodeOrdinal) + tTranslationX;
        tMappedLocations(1, nodeOrdinal) = aLocations(1, nodeOrdinal) + tTranslationY;
        tMappedLocations(2, nodeOrdinal) = aLocations(2, nodeOrdinal) + tTranslationZ;
    });

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
    });

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
    });

    return tElementWiseMap;
}

void check_for_missing_parent_elements(const Plato::OrdinalVector & aParentElements)
{
    Plato::OrdinalType tNumMissingParent(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, aParentElements.size()),
    KOKKOS_LAMBDA(const Plato::OrdinalType& aElemOrdinal, Plato::OrdinalType & aUpdate)
    {
        if ( aParentElements(aElemOrdinal) == -2 ) 
        {  
            Kokkos::atomic_increment(&aUpdate);
        }
    }, tNumMissingParent);
    if ( tNumMissingParent > 0 )
    {
        std::ostringstream tMsg;
        tMsg << "NO PARENT ELEMENT COULD BE FOUND FOR AT LEAST ONE CHILD NODE IN CONTACT PAIR. \n";
        ANALYZE_THROWERR(tMsg.str())
    }
}

}

}
