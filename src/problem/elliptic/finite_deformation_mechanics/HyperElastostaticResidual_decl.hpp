#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_HYPERELASTOSTATICRESIDUAL_DECL_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_HYPERELASTOSTATICRESIDUAL_DECL_H

#include <Teuchos_ParameterList.hpp>
#include <optional>

#include "boundary_conditions/BodyLoads.hpp"
#include "boundary_conditions/NaturalBCs.hpp"
#include "domain/Solutions.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "local_operations/optimization/ApplyWeighting.hpp"
#include "material/NeoHookeanModel.hpp"
#include "problem/elliptic/AbstractVectorFunction.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
/// @brief residual defining static hyperelastic physics.
/// This is implemented in a total Lagrangian approach, where the weak form of momentum balance is:
/// \int_{\Omega_0} \bm{P} : \bm{\nabla} \delta \bm{u} dV = \int_{\partial \Omega_0} \delta \bm{u} . \bm{T}^p dS +
/// \int_{\Omega_0} \delta \bm{u} . \bm{b}_0 dV
/// Here, \bm{u} is the unknown displacement field (trial solution),
/// \delta \bm{u} is the weighting function,
/// \bm{\nabla} is the gradient operator w.r.t the reference configuration,
/// \bm{P} is the first Piola-Kirchhoff stress tensor,
/// \bm{T}^p is the prescribed traction,
/// \bm{b}_0 is the prescribed body force.

/// @tparam EvaluationType struct containing automatic differentiation types for different evaluation cases
/// (e.g. Residual, Jacobian, GradientZ, GradientX)
/// @tparam IndicatorFunctionType pennalty function used for density-based methods
template <typename EvaluationType, typename IndicatorFunctionType>
class HyperElastostaticResidual : public EvaluationType::ElementType,
                                  public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
{
   private:
    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;
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
    HyperElastostaticResidual(const Plato::SpatialDomain& aSpatialDomain,
                              Plato::DataMap& aDataMap,
                              Teuchos::ParameterList& aProblemParams,
                              Teuchos::ParameterList& aPenaltyParams);

    /// @brief function to output solution data.
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions& aSolutions) const override final;

    /// @brief function to compute volumetric contributions the residual.
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                  Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                  Plato::Scalar aTimeStep = 0.0) const override final;

    /// @brief function to compute Natural Boundary Condition contributions for the residual.
    void evaluate_boundary(const Plato::SpatialModel& aSpatialModel,
                           const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                           const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                           const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                           Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                           Plato::Scalar aTimeStep = 0.0) const override final;

   private:
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, 1, IndicatorFunctionType> mApplyWeighting;
    std::optional<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
    std::optional<Plato::NaturalBCs<ElementType>> mBoundaryLoads;
    composable_function_objects::material::NeoHookeanParameters mNeoHookeanParameters;
    std::vector<std::string> mPlotTable;
};
}  // namespace plato::elliptic::finite_deformation_mechanics

#endif
