#pragma once

#include "local_operations/optimization/ApplyWeighting.hpp"
#include "material/ThermoelasticMaterial.hpp"
#include "problem/parabolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
/**
 * \brief Compute internal thermo-elastic energy criterion, given by
 *                  /f$ f(z) = u^{T}K_u(z)u + T^{T}K_t(z)T /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
 **********************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
class InternalThermoelasticEnergy : public EvaluationType::ElementType,
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

    using FunctionBaseType = Plato::Parabolic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mSpatialDomain;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyFluxWeighting;

    std::vector<std::string> mPlottable;

   public:
    /**************************************************************************/
    InternalThermoelasticEnergy(const plato::domain::SpatialDomain& aSpatialDomain,
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
// class InternalThermoelasticEnergy

}  // namespace Parabolic

}  // namespace Plato
