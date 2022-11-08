#pragma once

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"

#include <Teuchos_ParameterList.hpp>
#include <string>

namespace Plato
{

namespace Contact
{

class ContactSurface
{
public:
    ContactSurface();

    void
    initialize
    (const Teuchos::ParameterList & aParams,
    Plato::Mesh                    aMesh);

    void
    addParentData
    (const Plato::OrdinalVector     & aParentElements,
     const Plato::OrdinalVector     & aElementWiseChildMap,
     const Plato::ScalarMultiVector & aMappedChildNodeLocations);

    std::string
    childSideSet() { return mChildSideSet; }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    childNodes() { return mChildNodes; }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    childElements() { return mChildElements; }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    childFaceLocalNodes() { return mChildFaceLocalNodes; }

    std::string
    parentBlock() { return mParentBlock; }

    Plato::OrdinalVector
    parentElements();

    Plato::OrdinalVector
    elementWiseChildMap();

    Plato::ScalarMultiVector
    mappedChildNodeLocations();

private:
    std::string mChildSideSet;
    Plato::OrdinalVectorT<const Plato::OrdinalType> mChildNodes;
    Plato::OrdinalVectorT<const Plato::OrdinalType> mChildElements;
    Plato::OrdinalVectorT<const Plato::OrdinalType> mChildFaceLocalNodes;
    std::string mParentBlock;

    bool mHasParentData;
    Plato::OrdinalVector mParentElements;
    Plato::OrdinalVector mElementWiseChildMap;
    Plato::ScalarMultiVector mMappedChildNodeLocations;
};

struct ContactPair
{
    ContactSurface surfaceA;
    ContactSurface surfaceB;
    Teuchos::Array<Plato::Scalar> initialGap;
};

}

}
