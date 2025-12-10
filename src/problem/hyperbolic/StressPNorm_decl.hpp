#pragma once

#include "local_operations/constitutive/TensorNormBase.hpp"
#include "local_operations/optimization/ApplyWeighting.hpp"
#include "material/ElasticModelFactory.hpp"
#include "problem/hyperbolic/AbstractScalarFunction.hpp"
#include "problem/hyperbolic/EvaluationTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

template <typename EvaluationType, typename IndicatorFunctionType>
class StressPNorm : public EvaluationType::ElementType, public Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>
{
   private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    using FunctionBaseType = typename Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mSpatialDomain;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms, EvaluationType>> mNorm;

    std::string mFuncString = "1.0";

    Plato::ScalarMultiVector mFxnValues;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

   public:
    StressPNorm(const plato::domain::SpatialDomain& aSpatialDomain,
                Plato::DataMap& aDataMap,
                Teuchos::ParameterList& aProblemParams,
                Teuchos::ParameterList& aPenaltyParams,
                const std::string& aFunctionName);

    void evaluate_conditional(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                              const Plato::ScalarMultiVectorT<StateDotScalarType>& aStateDot,
                              const Plato::ScalarMultiVectorT<StateDotDotScalarType>& aStateDotDot,
                              const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                              const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                              Plato::ScalarVectorT<ResultScalarType>& aResult,
                              Plato::Scalar aTimeStep = 0.0) const override;

    void postEvaluate(Plato::ScalarVector resultVector, Plato::Scalar resultScalar) override;

    void postEvaluate(Plato::Scalar& resultValue) override;
};

}  // namespace Hyperbolic

}  // namespace Plato
