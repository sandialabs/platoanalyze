#pragma once

#include "AbstractSurfaceDisplacement.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato
{

namespace Contact
{

template <typename EvaluationType,
          Plato::OrdinalType NumDofsPerNode = EvaluationType::ElementType::mNumSpatialDims>
class SurfaceDisplacement : 
    public AbstractSurfaceDisplacement<EvaluationType>
 {

private: 
    using ElementType = typename EvaluationType::ElementType;
    using InStateT    = typename EvaluationType::StateScalarType;  
    using OutStateT   = typename EvaluationType::ResultScalarType; 

    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerFace;

public:
    SurfaceDisplacement
     (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aSideSetElements,
      const Plato::OrdinalVectorT<const Plato::OrdinalType> & aSideSetLocalNodes,
      Plato::Scalar                                           aScale = 1.0) :
     AbstractSurfaceDisplacement<EvaluationType>(aScale),
     mSideSetElements(aSideSetElements),
     mSideSetLocalNodes(aSideSetLocalNodes)
    {
    }

    KOKKOS_INLINE_FUNCTION void
    operator()
    (Plato::OrdinalType                               aCellOrdinal, 
     const Plato::Array<mNumNodesPerFace>           & aBasisFunctions,
     const Plato::ScalarMultiVectorT<InStateT>      & aState,
           Plato::Array<mNumSpatialDims, OutStateT> & aSurfaceDisp) const override
    {
        auto tGlobalCellOrdinal = mSideSetElements(aCellOrdinal);

        auto tScale = this->mScale;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
        {
            aSurfaceDisp(tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerFace; tNodeIndex++)
            {
                Plato::OrdinalType tSurfaceNode = mSideSetLocalNodes(aCellOrdinal*mNumNodesPerFace + tNodeIndex);
                Plato::OrdinalType tCellDofIndex = NumDofsPerNode * tSurfaceNode + tDofIndex; 
                aSurfaceDisp(tDofIndex) += tScale * aBasisFunctions(tNodeIndex) * aState(tGlobalCellOrdinal, tCellDofIndex);
            }
        }
    }

private:
    Plato::OrdinalVectorT<const Plato::OrdinalType> mSideSetElements;
    Plato::OrdinalVectorT<const Plato::OrdinalType> mSideSetLocalNodes;

};

}

}