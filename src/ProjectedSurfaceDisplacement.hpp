#pragma once

#include "AbstractSurfaceDisplacement.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoMesh.hpp"

namespace Plato
{

namespace Contact
{

template <typename EvaluationType,
          Plato::OrdinalType NumDofsPerNode = EvaluationType::ElementType::mNumSpatialDims>
class ProjectedSurfaceDisplacement : 
    public AbstractSurfaceDisplacement<EvaluationType>
{
private: 
    using ElementType = typename EvaluationType::ElementType;
    using InStateT    = typename EvaluationType::StateScalarType;  
    using OutStateT   = typename EvaluationType::ResultScalarType; 

    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerFace;
    using ElementType::mNumNodesPerCell;

public:
    ProjectedSurfaceDisplacement
    (const Plato::OrdinalVectorT<Plato::OrdinalType> & aParentElements,
     const Plato::ScalarMultiVectorT<Plato::Scalar>  & aMappedLocations,
     const Plato::OrdinalVector                      & aChildNodeOrdMap,
           Plato::Mesh                                 aMesh,
           Plato::Scalar                               aScale = 1.0) :
     AbstractSurfaceDisplacement<EvaluationType>(aScale),
     mParentElements(aParentElements),
     mMappedLocations(aMappedLocations),
     mChildNodeOrdMap(aChildNodeOrdMap),
     mGetBasis(aMesh),
     mInterpolateFromNodal(),
     mChildNode(0)
    {
    }

    KOKKOS_INLINE_FUNCTION void
    operator()
    (Plato::OrdinalType                               aCellOrdinal, 
     const Plato::Array<mNumNodesPerFace>           & aBasisFunctions,
     const Plato::ScalarMultiVectorT<InStateT>      & aState,
           Plato::Array<mNumSpatialDims, OutStateT> & aSurfaceDisp) const override
    {
        auto tLocalChildNodeOrd = mChildNodeOrdMap(aCellOrdinal*mNumNodesPerFace + mChildNode);
        auto tParentElement = mParentElements(tLocalChildNodeOrd);

        Plato::Array<mNumSpatialDims, Plato::Scalar> tInPoint(0.0);
        for(Plato::OrdinalType iDim=0; iDim<mNumSpatialDims; iDim++)
        {
            tInPoint(iDim) = mMappedLocations(iDim, tLocalChildNodeOrd);
        }

        Plato::Array<mNumNodesPerCell, Plato::Scalar> tBasis(0.0); // config scalar type
        mGetBasis(tParentElement, tInPoint, tBasis);

        mInterpolateFromNodal(tParentElement, tBasis, aState, aSurfaceDisp);

        auto tScale = this->mScale;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
        {
            aSurfaceDisp(tDofIndex) *= tScale * aBasisFunctions(mChildNode);
        }
    }

    void setChildNode(Plato::OrdinalType aChildNode)
    {
        mChildNode = aChildNode;
    }

private: 
    Plato::OrdinalVectorT<Plato::OrdinalType>             mParentElements;
    Plato::ScalarMultiVectorT<Plato::Scalar>              mMappedLocations;
    Plato::OrdinalVector                                  mChildNodeOrdMap;
    Plato::OrdinalType                                    mChildNode;
    Plato::Geometry::GetBasis<ElementType, Plato::Scalar> mGetBasis;
    Plato::InterpolateFromNodal<ElementType, NumDofsPerNode, /*offset=*/0, mNumSpatialDims> mInterpolateFromNodal;

};

}

}