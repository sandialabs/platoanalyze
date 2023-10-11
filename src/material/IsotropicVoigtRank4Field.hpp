#pragma once

#include "material/Rank4Field.hpp"
#include "material/ScalarExpression.hpp"

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

namespace Plato
{

template<typename EvaluationType>
class IsotropicVoigtRank4Field : public Rank4Field<EvaluationType>
{
protected:
    using ElementType = typename EvaluationType::ElementType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
  
public:
    IsotropicVoigtRank4Field(const Teuchos::ParameterList& aParams) : 
    Rank4Field<EvaluationType>() 
    {
        this->mStiffnessTensorProperties.clear();
        this->mStiffnessTensorProperties["Youngs Modulus"] = Plato::ScalarExpression<EvaluationType>("Youngs Modulus", aParams);
        this->mStiffnessTensorProperties["Poissons Ratio"] = Plato::ScalarExpression<EvaluationType>("Poissons Ratio", aParams);
    }

    Plato::ScalarArray4DT<KineticsScalarType>
    operator()(const Plato::ScalarMultiVectorT<ControlScalarType>& aLocalControl) override
    {
        Plato::OrdinalType tNumCells = aLocalControl.extent(0);
        Plato::OrdinalType tNumPoints = ElementType::getCubWeights().size();

        Plato::ScalarVectorT<ControlScalarType> tIndependentVariable("density", tNumCells*tNumPoints);
        this->calculateIndependentVariable(aLocalControl, tIndependentVariable);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementYoungsModulus = this->getTensorProperty("Youngs Modulus")(tIndependentVariable);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementPoissonsRatio = this->getTensorProperty("Poissons Ratio")(tIndependentVariable);
      
        Plato::ScalarArray4DT<KineticsScalarType> tStiffness("stiffness", tNumCells, tNumPoints, ElementType::mNumVoigtTerms, ElementType::mNumVoigtTerms);   
        Kokkos::parallel_for("compute stiffness tensor", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),      
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            const auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;
            const auto tCurYoungsModulus = tElementYoungsModulus(tEntryOrdinal, 0);
            const auto tCurPoissonsRatio = tElementPoissonsRatio(tEntryOrdinal, 0);
            const auto tCoeff = tCurYoungsModulus / ((1.0 + tCurPoissonsRatio) * (1.0 - 2.0 * tCurPoissonsRatio));
            for(int k=0; k<ElementType::mNumSpatialDims; ++k)
            {
                for(int m=0; m<ElementType::mNumSpatialDims; ++m)
                {
                    if(k==m)
                    {
                        tStiffness(iCellOrdinal, iGpOrdinal, k, m) = tCoeff * (1.0 - tCurPoissonsRatio);
                    }
                    else
                    {
                        tStiffness(iCellOrdinal, iGpOrdinal, k, m) = tCoeff * tCurPoissonsRatio;
                    }
                }
            }
            const int tNumShearTerms = ElementType::mNumSpatialDims*(ElementType::mNumSpatialDims-1)/2;
            for(int m=0; m<tNumShearTerms; ++m)
            {
                tStiffness(iCellOrdinal, iGpOrdinal, ElementType::mNumSpatialDims+m, ElementType::mNumSpatialDims+m) = 
                               1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tCurPoissonsRatio);
            }
        });

        return tStiffness;
    }
};

}