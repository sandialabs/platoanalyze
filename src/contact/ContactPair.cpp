#include "contact/ContactPair.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Contact
{

ContactSurface::ContactSurface() : mHasParentData(false)
{
}

void
ContactSurface::initialize
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh)
{
    if (!aParams.isType<std::string>("Child Sideset"))
        ANALYZE_THROWERR("Child Sideset was not provided in contact pair")

    std::string tSideSet = aParams.get<std::string>("Child Sideset");
    mChildSideSet        = tSideSet;
    mChildNodes          = aMesh->GetNodeSetNodes(tSideSet);
    mChildElements       = aMesh->GetSideSetElements(tSideSet);
    mChildFaceLocalNodes = aMesh->GetSideSetLocalNodes(tSideSet);

    if (!aParams.isType<std::string>("Parent Block"))
        ANALYZE_THROWERR("Parent Block was not provided in contact pair")

    mParentBlock = aParams.get<std::string>("Parent Block");
}

void
ContactSurface::addParentData
(const Plato::OrdinalVector     & aParentElements,
 const Plato::OrdinalVector     & aElementWiseChildMap,
 const Plato::ScalarMultiVector & aMappedChildNodeLocations)
{
    if (!mHasParentData)
    {
        mParentElements           = aParentElements;
        mElementWiseChildMap      = aElementWiseChildMap;
        mMappedChildNodeLocations = aMappedChildNodeLocations;

        mHasParentData = true;
    }
}

Plato::OrdinalVector
ContactSurface::parentElements() const
{ 
    if (!mHasParentData)
        ANALYZE_THROWERR("In ContactSurface class: Attempting to access parent data before it is assigned.")

    return mParentElements; 
}

Plato::OrdinalVector
ContactSurface::elementWiseChildMap() const
{ 
    if (!mHasParentData)
        ANALYZE_THROWERR("In ContactSurface class: Attempting to access parent data before it is assigned.")

    return mElementWiseChildMap; 
}

Plato::ScalarMultiVector
ContactSurface::mappedChildNodeLocations() const
{ 
    if (!mHasParentData)
        ANALYZE_THROWERR("In ContactSurface class: Attempting to access parent data before it is assigned.")

    return mMappedChildNodeLocations; 
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
        Kokkos::parallel_for("store child nodes and parent elements", Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
        {
            aChildNodes(tOffset + nodeOrdinal) = tChildNodes(nodeOrdinal);
            aParentElements(tOffset + nodeOrdinal) = tParentElements(nodeOrdinal);
        });
        tOffset += tNumNodes;

        tChildNodes = tPair.surfaceB.childNodes();
        tParentElements = tPair.surfaceB.parentElements();
        tNumNodes = tChildNodes.size();
        auto tScaledGap = scale_initial_gap(tPair.initialGap, -1.0);
        Kokkos::parallel_for("store child nodes and parent elements", Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
        {
            aChildNodes(tOffset + nodeOrdinal) = tChildNodes(nodeOrdinal);
            aParentElements(tOffset + nodeOrdinal) = tParentElements(nodeOrdinal);
        });
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
    });

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
