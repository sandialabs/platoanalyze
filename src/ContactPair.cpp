#include "ContactPair.hpp"
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
ContactSurface::parentElements() 
{ 
    if (!mHasParentData)
        ANALYZE_THROWERR("In ContactSurface class: Attempting to access parent data before it is assigned.")

    return mParentElements; 
}

Plato::OrdinalVector
ContactSurface::elementWiseChildMap() 
{ 
    if (!mHasParentData)
        ANALYZE_THROWERR("In ContactSurface class: Attempting to access parent data before it is assigned.")

    return mElementWiseChildMap; 
}

Plato::ScalarMultiVector
ContactSurface::mappedChildNodeLocations() 
{ 
    if (!mHasParentData)
        ANALYZE_THROWERR("In ContactSurface class: Attempting to access parent data before it is assigned.")

    return mMappedChildNodeLocations; 
}

}

}
