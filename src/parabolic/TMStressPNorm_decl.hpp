#pragma once

#include "ApplyWeighting.hpp"
#include "TensorPNorm.hpp"
#include "ThermoelasticMaterial.hpp"
#include "parabolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
class TMStressPNorm : public EvaluationType::ElementType,
                      public Plato::Parabolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
   private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumDofsPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    static constexpr int TDofOffset = mNumSpatialDims;

    using FunctionBaseType = typename Plato::Parabolic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mSpatialDomain;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms, EvaluationType>> mNorm;

    std::string mFuncString = "1.0";

   public:
    /**************************************************************************/
    TMStressPNorm(const Plato::SpatialDomain& aSpatialDomain,
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

    /**************************************************************************/
    void postEvaluate(Plato::ScalarVector resultVector, Plato::Scalar resultScalar) override;
    /**************************************************************************/

    /**************************************************************************/
    void postEvaluate(Plato::Scalar& resultValue) override;
    /**************************************************************************/
};
// class TMStressPNorm

}  // namespace Parabolic

}  // namespace Plato
