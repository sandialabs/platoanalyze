#pragma once

#include "material/ScalarExpression.hpp"
#include "InterpolateFromNodal.hpp"

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

#include <string>
#include <map>

namespace Plato
{

template<typename EvaluationType>
class Rank4Field
{
protected:
    using ElementType = typename EvaluationType::ElementType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
  
public:
    Rank4Field() = default;

    virtual ~Rank4Field() = default;

    Rank4Field(const Rank4Field& aField) = delete;

    Rank4Field(Rank4Field&& aField) = delete;

    Rank4Field&
    operator=(const Rank4Field& aField) = delete;

    Rank4Field&
    operator=(Rank4Field&& aField) = delete;

    Plato::ScalarExpression<EvaluationType>
    getTensorProperty(const std::string &aName) const 
    { 
        if (mStiffnessTensorProperties.count(aName) == 0)
          ANALYZE_THROWERR("Expression with name '" + aName + "' not found in Rank4Field");
        return mStiffnessTensorProperties.at(aName);
    }

    virtual Plato::ScalarArray4DT<KineticsScalarType>
    operator()(const Plato::ScalarMultiVectorT<ControlScalarType>& aLocalControl)  
    { 
        Plato::ScalarArray4DT<KineticsScalarType> tStiffness("stiffness", 0,0,0,0);   
        return tStiffness;
    }

    void 
    calculateIndependentVariable
    (const Plato::ScalarMultiVectorT<ControlScalarType>& aLocalControl,
           Plato::ScalarVectorT<ControlScalarType>& aIndependentVariable)
    {
        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        Plato::OrdinalType tNumPoints = tCubWeights.size();
        Plato::OrdinalType tNumCells = aLocalControl.extent(0);
        Plato::InterpolateFromNodal<ElementType, 1, 0> tInterpolateFromNodal;

        Kokkos::parallel_for("compute independent variable", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            // Calculate the node-averaged independent variable for the element/cell
            auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;
            aIndependentVariable(tEntryOrdinal) = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aLocalControl);
        });
    }

protected:
    std::map<std::string, Plato::ScalarExpression<EvaluationType>> mStiffnessTensorProperties;
};

}