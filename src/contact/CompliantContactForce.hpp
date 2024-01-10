#pragma once

#include "contact/AbstractContactForce.hpp"
#include "PlatoStaticsTypes.hpp"

#include <Teuchos_Array.hpp>

namespace Plato
{

namespace Contact
{

template<typename EvaluationType>
class CompliantContactForce : public AbstractContactForce<EvaluationType>
{
private:
    using ElementType = typename EvaluationType::ElementType;
    using StateType  = typename EvaluationType::StateScalarType;  
    using ConfigType = typename EvaluationType::ConfigScalarType; 
    using ResultType = typename EvaluationType::ResultScalarType; 

public:
    CompliantContactForce(const Teuchos::Array<Plato::Scalar> & aPenaltyValue)
    {
        for(Plato::OrdinalType iDim=0; iDim < ElementType::mNumSpatialDims; iDim++)
            mPenaltyValue(iDim) = aPenaltyValue[iDim];
    }

    void
    operator()
    (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds,
     const Plato::OrdinalVectorT<const Plato::OrdinalType> & aLocalNodeOrds,
     const Plato::ScalarArray3DT<StateType>  & aState,
     const Plato::ScalarArray3DT<ConfigType> & aConfig,
           Plato::ScalarArray3DT<ResultType> & aResult) const override
    {
        Plato::OrdinalType tNumCells = aElementOrds.size();

        auto tCubatureWeights = ElementType::Face::getCubWeights();
        Plato::OrdinalType tNumPoints = tCubatureWeights.size();
        
        auto tPenaltyValue = mPenaltyValue;
        Kokkos::parallel_for("apply contact penalization and projection",  Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            for(Plato::OrdinalType iDim = 0; iDim < ElementType::mNumSpatialDims; iDim++)
                aResult(iCellOrdinal, iGPOrdinal, iDim) = tPenaltyValue(iDim) * aState(iCellOrdinal, iGPOrdinal, iDim);
        });

    }

private:
    Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> mPenaltyValue;
};

}

}
