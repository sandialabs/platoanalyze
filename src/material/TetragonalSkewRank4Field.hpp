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
class TetragonalSkewRank4Field : public Rank4Field<EvaluationType>
{
protected:
    using ElementType = typename EvaluationType::ElementType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
  
public:
    TetragonalSkewRank4Field(const Teuchos::ParameterList& aParams) : 
    Rank4Field<EvaluationType>() 
    {
        this->mStiffnessTensorProperties.clear();
        this->mStiffnessTensorProperties["Mu"] = Plato::ScalarExpression<EvaluationType>("Mu", aParams);
    }

    Plato::ScalarArray4DT<KineticsScalarType>
    operator()(const Plato::ScalarMultiVectorT<ControlScalarType>& aLocalControl) override
    {
        Plato::OrdinalType tNumCells = aLocalControl.extent(0);
        Plato::OrdinalType tNumPoints = ElementType::getCubWeights().size();

        Plato::ScalarVectorT<ControlScalarType> tIndependentVariable("density", tNumCells*tNumPoints);
        this->calculateIndependentVariable(aLocalControl, tIndependentVariable);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementMu = this->getTensorProperty("Mu")(tIndependentVariable);
      
        Plato::ScalarArray4DT<KineticsScalarType> tStiffness("stiffness", tNumCells, tNumPoints, mNumSkwTerms, mNumSkwTerms);   
        Kokkos::parallel_for("compute stiffness tensor", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),      
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            const auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;
            const auto tCurMu = tElementMu(tEntryOrdinal, 0);
            for(int k=0; k<mNumSkwTerms; ++k)
            {
                tStiffness(iCellOrdinal, iGpOrdinal, k, k) = tCurMu;
            }
        });

        return tStiffness;
    }

private:
    static constexpr auto mNumSkwTerms = (ElementType::mNumSpatialDims == 3) ? 3 :
                                        ((ElementType::mNumSpatialDims == 2) ? 1 :
                                        (((ElementType::mNumSpatialDims == 1) ? 1 : 0)));
};

}