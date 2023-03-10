#pragma once

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Contact
{

class UpdateGraphForContact
{
public:
    UpdateGraphForContact(Plato::Mesh aMesh);

    void 
    createNodeNodeGraph
    (const Plato::OrdinalVector & aChildNodes,
     const Plato::OrdinalVector & aParentElements);

    void
    NodeNodeGraph
    (Plato::OrdinalVector & aOffsetMap,
     Plato::OrdinalVector & aNodeOrds) const;

    void
    NodeNodeGraphTranspose
    (Plato::OrdinalVector & aOffsetMap,
     Plato::OrdinalVector & aNodeOrds) const;

    Plato::OrdinalType 
    extractChildNodeOffsets(const Plato::OrdinalVector & aChildNodes);

    void 
    storeUniqueParentNodeContributions
    (const Plato::OrdinalVector & aChildNodes, 
     const Plato::OrdinalVector & aParentElements);

    Plato::OrdinalType 
    updateOffsetMap();

    void 
    updateNodeOrds();

    void 
    countNonzerosForTranspose(Plato::OrdinalVector & aOffsetMap) const;

    Plato::OrdinalType 
    constructTransposeOffsetMap(Plato::OrdinalVector & aOffsetMap) const;

    void
    constructTransposeNodeOrds
    (const Plato::OrdinalVector & aOffsetMap,
           Plato::OrdinalVector & aNodeOrds) const;

private:
    Plato::OrdinalVectorT<const Plato::OrdinalType> mConnectivity;

    Plato::OrdinalType mNumTotalNodes;
    Plato::OrdinalType mNumNodesPerElement;

    Plato::OrdinalVector mMarkedChildNodes;

    Plato::OrdinalVector mOffsetMap;
    Plato::OrdinalVector mNodeOrds;

    Plato::OrdinalVector mFullOffsetMap;
    Plato::OrdinalVector mFullNodeOrds;

    Plato::OrdinalVector mChildOffsetMap;
    Plato::OrdinalVector mNumConnectedNodes;
    Plato::OrdinalVector mAllGraphOrdinals;
};

}

}
