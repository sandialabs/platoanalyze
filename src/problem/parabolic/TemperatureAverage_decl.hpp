#pragma once

#include "local_operations/optimization/ApplyWeighting.hpp"
#include "problem/parabolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
class TemperatureAverage : public EvaluationType::ElementType,
                           public Plato::Parabolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
   private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumDofsPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Parabolic::AbstractScalarFunction<EvaluationType>;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumDofsPerNode, IndicatorFunctionType> mApplyWeighting;

   public:
    /**************************************************************************/
    TemperatureAverage(const plato::domain::SpatialDomain& aSpatialDomain,
                       Plato::DataMap& aDataMap,
                       Teuchos::ParameterList& aProblemParams,
                       Teuchos::ParameterList& aPenaltyParams,
                       const std::string& aFunctionName);

    /**************************************************************************/
    void evaluate_conditional(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                              const Plato::ScalarMultiVectorT<StateDotScalarType>& aStateDot,
                              const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                              const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                              Plato::ScalarVectorT<ResultScalarType>& aResult,
                              Plato::Scalar aTimeStep = 0.0) const override;
};
// class TemperatureAverage

}  // namespace Parabolic

}  // namespace Plato
