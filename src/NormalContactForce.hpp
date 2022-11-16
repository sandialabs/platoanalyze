#pragma once

#include "AbstractContactForce.hpp"
#include "PlatoStaticsTypes.hpp"
#include "WeightedNormalVector.hpp"
#include "SurfaceArea.hpp"

#include <Teuchos_Array.hpp>

namespace Plato
{

namespace Contact
{

template<typename EvaluationType>
class NormalContactForce : public AbstractContactForce<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;
    using StateType  = typename EvaluationType::StateScalarType;  
    using ConfigType = typename EvaluationType::ConfigScalarType; 
    using ResultType = typename EvaluationType::ResultScalarType; 

public:
    NormalContactForce(const Teuchos::Array<Plato::Scalar> & aPenaltyValue)
    {
        if (aPenaltyValue.size() != 1)
            ANALYZE_THROWERR("More than 1 penalty value specified for Contact Pair of 'normal' Type");

        mPenaltyValue = aPenaltyValue[0];
    }

    void
    operator()
    (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds,
     const Plato::OrdinalVectorT<const Plato::OrdinalType> & aLocalNodeOrds,
     const Plato::ScalarArray3DT<StateType>  & aState,
     const Plato::ScalarArray3DT<ConfigType> & aConfig,
           Plato::ScalarArray3DT<ResultType> & aResult) const override
    {
        auto tNumCells = aElementOrds.size();

        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tCubatureWeights = ElementType::Face::getCubWeights();
        auto tNumPoints = tCubatureWeights.size();

        Plato::WeightedNormalVector<ElementType> weightedNormalVector;
        Plato::SurfaceArea<ElementType> surfaceArea;

        auto tPenaltyValue = mPenaltyValue;
        Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tGlobalCellOrdinal = aElementOrds(iCellOrdinal);

            auto tCubaturePoint = tCubaturePoints(iGPOrdinal);
            auto tBasisGrads = ElementType::Face::basisGrads(tCubaturePoint);

            Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
            for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
            {
                tLocalNodeOrds(tNodeOrd) = aLocalNodeOrds(iCellOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
            }

            Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tWeightedNormalVec;
            weightedNormalVector(tGlobalCellOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tWeightedNormalVec);

            Plato::Scalar tSurfaceArea(0.0);
            surfaceArea(tGlobalCellOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tSurfaceArea);

            Plato::Scalar tProduct(0.0);
            for(Plato::OrdinalType iDim = 0; iDim < ElementType::mNumSpatialDims; iDim++)
                tProduct += aState(iCellOrdinal, iGPOrdinal, iDim) * tWeightedNormalVec(iDim) / tSurfaceArea;

            for(Plato::OrdinalType iDim = 0; iDim < ElementType::mNumSpatialDims; iDim++)
                aResult(iCellOrdinal, iGPOrdinal, iDim) = tPenaltyValue * tProduct * tWeightedNormalVec(iDim) / tSurfaceArea;

        }, "apply contact penalization and projection");

    }

private:
    Plato::Scalar mPenaltyValue;
};

}

}
