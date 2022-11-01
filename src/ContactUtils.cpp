#include "ContactUtils.hpp"

namespace Plato
{

ContactPair parseContactPair
(const Teuchos::ParameterList            & aParams,
 Plato::Mesh                               aMesh)
{
    ContactPair tContactPair;

    if (!aParams.isType<std::string>("Side A Child Sideset"))
        ANALYZE_THROWERR("Side A Child Sideset was not provided in contact pair")

    std::string tSideSetA = aParams.get<std::string>("Side A Child Sideset");
    tContactPair.childNodesA = aMesh->GetNodeSetNodes(tSideSetA);

    if (!aParams.isType<std::string>("Side B Child Sideset"))
        ANALYZE_THROWERR("Side B Child Sideset was not provided in contact pair")

    std::string tSideSetB = aParams.get<std::string>("Side B Child Sideset");
    tContactPair.childNodesB = aMesh->GetNodeSetNodes(tSideSetB);
    
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

Plato::SpatialDomain getDomain
(const std::string                       & aDomainName,
 const std::vector<Plato::SpatialDomain> & aDomains)
{
    for(auto& tDomain : aDomains)
    {
        auto tName = tDomain.getElementBlockName();
        if( tName == aDomainName )
            return tDomain;
    }
    std::string tMsg = "Block with name " + aDomainName + " provided in getDomain does not correspond to an element block name in spatial model";
    ANALYZE_THROWERR(tMsg)
}

Plato::ScalarMultiVector computeNodeLocations
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

Plato::ScalarMultiVector mapNodeLocations
(const Plato::ScalarMultiVector      & aLocations,
 const Teuchos::Array<Plato::Scalar> & aTranslation,
 Plato::Scalar                         aScale)
{
    static constexpr int tSpaceDim = 3;
    if (aTranslation.size() != tSpaceDim || aLocations.extent(0) != tSpaceDim)
        ANALYZE_THROWERR("In ContactUtils mapNodeLocations, an incorrect dimension is given. Only 3 dimensions are supported.")
    
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

}
// namespace Plato
