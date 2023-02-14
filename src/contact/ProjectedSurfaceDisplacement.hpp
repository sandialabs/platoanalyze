#pragma once

#include "contact/AbstractSurfaceDisplacement.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoMesh.hpp"
#include "InterpolateFromNodal.hpp"
#include "Plato_MeshMapUtils.hpp"

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
    using OutStateT   = typename EvaluationType::StateScalarType; 

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
     mChildNode(0)
    {
    }

    void
    operator()
    (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds,
     const Plato::ScalarMultiVectorT<InStateT>             & aState,
           Plato::ScalarArray3DT<OutStateT>                & aSurfaceDisp) const override
    {
        Plato::OrdinalType tNumFaces = aElementOrds.size();

        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tCubatureWeights = ElementType::Face::getCubWeights();
        Plato::OrdinalType tNumPoints = tCubatureWeights.size();

        Plato::InterpolateFromNodal<ElementType, NumDofsPerNode, /*offset=*/0, ElementType::mNumSpatialDims> tInterpolateFromNodal;

        auto tChildNode = mChildNode;
        auto tScale = this->mScale;
        auto& tChildNodeOrdMap = mChildNodeOrdMap;
        auto& tParentElements = mParentElements;
        auto& tMappedLocations = mMappedLocations;
        auto& tGetBasis = mGetBasis;

        Kokkos::parallel_for("projected surface displacement", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tLocalChildNodeOrd = tChildNodeOrdMap(iCellOrdinal*ElementType::mNumNodesPerFace + tChildNode);
            auto tParentElement = tParentElements(tLocalChildNodeOrd);

            Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);
            for(Plato::OrdinalType iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
            {
                tInPoint(iDim) = tMappedLocations(iDim, tLocalChildNodeOrd);
            }

            Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tBasis(0.0); // config scalar type
            tGetBasis(tParentElement, tInPoint, tBasis);

            Plato::Array<ElementType::mNumSpatialDims, OutStateT> tSurfaceDisp(0.0); // config scalar type
            tInterpolateFromNodal(tParentElement, tBasis, aState, tSurfaceDisp);

            auto tCubaturePoint = tCubaturePoints(iGPOrdinal);
            auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);

            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
            {
                aSurfaceDisp(iCellOrdinal, iGPOrdinal, tDofIndex) = tScale * tBasisValues(tChildNode) * tSurfaceDisp(tDofIndex);
            }

        });
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

};

}

}