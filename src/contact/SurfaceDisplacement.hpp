#pragma once

#include "contact/AbstractSurfaceDisplacement.hpp"
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
    using OutStateT   = typename EvaluationType::StateScalarType; 

public:
    SurfaceDisplacement
    (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aSideSetLocalNodes,
     Plato::Scalar                                          aScale = 1.0) :
     AbstractSurfaceDisplacement<EvaluationType>(aScale),
     mSideSetLocalNodes(aSideSetLocalNodes)
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

        auto tScale = this->mScale;
        auto& tSideSetLocalNodes = mSideSetLocalNodes;

        Kokkos::parallel_for("surface displacement", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tGlobalCellOrdinal = aElementOrds(iCellOrdinal);

            auto tCubaturePoint = tCubaturePoints(iGPOrdinal);
            auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);

            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < NumDofsPerNode; tDofIndex++)
            {
                aSurfaceDisp(iCellOrdinal, iGPOrdinal, tDofIndex) = 0.0;
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerFace; tNodeIndex++)
                {
                    Plato::OrdinalType tFaceNode = tSideSetLocalNodes(iCellOrdinal*ElementType::mNumNodesPerFace + tNodeIndex);
                    Plato::OrdinalType tCellDofIndex = NumDofsPerNode * tFaceNode + tDofIndex; 
                    aSurfaceDisp(iCellOrdinal, iGPOrdinal, tDofIndex) += tScale * tBasisValues(tNodeIndex) * aState(tGlobalCellOrdinal, tCellDofIndex);
                }
            }

        });
    }

private:
    Plato::OrdinalVectorT<const Plato::OrdinalType> mSideSetLocalNodes;

};

}

}