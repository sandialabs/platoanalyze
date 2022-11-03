#pragma once

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"

#include <Teuchos_RCP.hpp>

namespace Plato
{

namespace Contact
{

class UpdateGraphForContact
{
public:
    UpdateGraphForContact
    (Plato::Mesh                  aMesh,
     const Plato::OrdinalVector & aChildNodes,
     const Plato::OrdinalVector & aParentElements);

    Teuchos::RCP<Plato::CrsMatrixType> 
    operator()
    (Teuchos::RCP<Plato::CrsMatrixType> aMatrix);

    Plato::OrdinalType 
    extractChildNodeOffsets(const Plato::OrdinalVector & aOffsetMap);

    void 
    storeUniqueParentNodeContributions
    (const Plato::OrdinalVector & aOffsetMap, 
     const Plato::OrdinalVector & aNodeOrds);

    Plato::OrdinalType 
    updateOffsetMap(const Plato::OrdinalVector & aOffsetMap);

    void 
    updateNodeOrds
    (const Plato::OrdinalVector & aOffsetMap, 
    const Plato::OrdinalVector & aNodeOrds);

private:
    Plato::OrdinalVector mChildNodes;
    Plato::OrdinalVector mParentElements;

    Plato::OrdinalVectorT<const Plato::OrdinalType> mConnectivity;

    Plato::OrdinalType mNumTotalNodes;
    Plato::OrdinalType mNumNodesPerElement;

    Plato::OrdinalVector mChildOffsetMap;
    Plato::OrdinalVector mMarkedChildNodes;
    Plato::OrdinalVector mNumConnectedNodes;
    Plato::OrdinalVector mAllGraphOrdinals;

    Plato::OrdinalVector mFullOffsetMap;
    Plato::OrdinalVector mFullNodeOrds;
};

}

}
