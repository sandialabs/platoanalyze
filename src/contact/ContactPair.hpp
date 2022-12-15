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

    const std::string &
    childSideSet() const { return mChildSideSet; }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    childNodes() const { return mChildNodes; }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    childElements() const { return mChildElements; }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    childFaceLocalNodes() const { return mChildFaceLocalNodes; }

    const std::string &
    parentBlock() const { return mParentBlock; }

    Plato::OrdinalVector
    parentElements() const;

    Plato::OrdinalVector
    elementWiseChildMap() const;

    Plato::ScalarMultiVector
    mappedChildNodeLocations() const;

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
    Plato::Scalar searchTolerance;
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
