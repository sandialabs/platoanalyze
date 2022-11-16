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
    std::string penaltyType;
    Teuchos::Array<Plato::Scalar> penaltyValue;
};

Teuchos::Array<Plato::Scalar> 
scale_initial_gap
(const Teuchos::Array<Plato::Scalar> aGap,
 Plato::Scalar                       aScale);

Plato::OrdinalType count_total_child_nodes(const std::vector<ContactPair> & aPairs);

void populate_full_contact_arrays
(const std::vector<ContactPair> & aPairs,
       Plato::OrdinalVector     & aChildNodes,
       Plato::OrdinalVector     & aParentElements);

void check_for_repeated_child_nodes
(const Plato::OrdinalVector & aChildNodes,
       Plato::OrdinalType     aNumMeshNodes);

}

}
