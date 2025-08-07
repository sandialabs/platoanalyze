#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_STRAINENERGY_DECL_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_STRAINENERGY_DECL_H

#include <Teuchos_ParameterList.hpp>
#include <string>

#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
/// @brief criterion that computes the strain energy from a hyperelastic model.
/// @tparam EvaluationType struct containing automatic differentiation types for different evaluation cases
/// (e.g. Value, GradientU, GradientZ, GradientX)
/// @tparam IndicatorFunctionType pennalty function used for density-based methods
template <typename EvaluationType, typename IndicatorFunctionType>
class StrainEnergy : public EvaluationType::ElementType, public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
   private:
    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
    using FunctionBaseType::mSpatialDomain;

    using ElementType = typename EvaluationType::ElementType;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    static constexpr Plato::OrdinalType mTensorDim{3};

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

   public:
    StrainEnergy(const Plato::SpatialDomain& aSpatialDomain,
                 Plato::DataMap& aDataMap,
                 Teuchos::ParameterList& aProblemParams,
                 Teuchos::ParameterList& aPenaltyParams,
                 const std::string& aFunctionName);

    /// @brief compute criterion for every element and store the results in @param aResult.
    void evaluate_conditional(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                              const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                              const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                              Plato::ScalarVectorT<ResultScalarType>& aResult,
                              Plato::Scalar aTimeStep = 0.0) const override final;

   private:
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, 1, IndicatorFunctionType> mApplyWeighting;
    Plato::Scalar mKappa;
    Plato::Scalar mMu;
};
}  // namespace plato::elliptic::finite_deformation_mechanics
#endif
