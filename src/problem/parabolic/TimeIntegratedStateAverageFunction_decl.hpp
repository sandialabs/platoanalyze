#ifndef PLATO_PROBLEM_PARABOLIC_TIMEINTEGRATEDSTATEAVERAGEFUNCTION_DECL
#define PLATO_PROBLEM_PARABOLIC_TIMEINTEGRATEDSTATEAVERAGEFUNCTION_DECL

#include <Teuchos_ParameterList.hpp>
#include <string>
#include <string_view>

#include "domain/SpatialModel.hpp"
#include "domain/WorksetBase.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/parabolic/ScalarFunctionBase.hpp"

namespace plato::parabolic
{
[[nodiscard]] auto time_integrated_state_average_function_name() -> std::string_view;

/// @brief class for computing criteria consisting of the time integral of states averaged over a nodeset. Time
/// integration is carried out using the trapezoid rule.
template <typename PhysicsType>
class TimeIntegratedStateAverageFunction : public Plato::Parabolic::ScalarFunctionBase,
                                           public Plato::WorksetBase<typename PhysicsType::ElementType>
{
   private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;
    using Plato::WorksetBase<ElementType>::mNumNodes;
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;

   public:
    TimeIntegratedStateAverageFunction(const plato::domain::SpatialModel& aSpatialModel,
                                       Plato::DataMap& aDataMap,
                                       Teuchos::ParameterList& aProblemParams,
                                       const std::string& aName);

    ///@brief computes the criterion for a given solution @a aSolution and controls @a aControls.
    Plato::Scalar value(const Plato::Solutions& aSolution,
                        const Plato::ScalarVector& aControl,
                        const Plato::Scalar aTimeStep = 0.0) const override final;

    ///@brief computes the criterion gradient w.r.t. the state at time step @a aStepIndex for a given solution
    ///@a aSolution and controls @a aControls.
    Plato::ScalarVector gradient_u(const Plato::Solutions& aSolution,
                                   const Plato::ScalarVector& aControl,
                                   const Plato::OrdinalType aStepIndex,
                                   const Plato::Scalar aTimeStep = 0.0) const override final;

    ///@brief computes the criterion gradient w.r.t. the state time derivative at time step @a aStepIndex for a given
    /// solution @a aSolution and controls @a aControls.
    Plato::ScalarVector gradient_v(const Plato::Solutions& aSolution,
                                   const Plato::ScalarVector& aControl,
                                   const Plato::OrdinalType aStepIndex,
                                   const Plato::Scalar aTimeStep = 0.0) const override final;

    ///@brief computes the criterion gradient w.r.t. control for a given solution @a aSolution and controls
    /// @a aControls.
    Plato::ScalarVector gradient_z(const Plato::Solutions& aSolution,
                                   const Plato::ScalarVector& aControl,
                                   const Plato::Scalar aTimeStep = 0.0) const override final;

    ///@brief computes the criterion gradient w.r.t. nodal coordinates for a given solution @a aSolution and controls
    /// @a aControls.
    Plato::ScalarVector gradient_x(const Plato::Solutions& aSolution,
                                   const Plato::ScalarVector& aControl,
                                   const Plato::Scalar aTimeStep = 0.0) const override final;

    ///@brief returns the name of the criterion being computed
    std::string name() const override final;

   private:
    std::string mName;
    plato::domain::SpatialModel mSpatialModel;
    std::string mNodeSet;
    Plato::OrdinalType mStateComponent;
};
}  // namespace plato::parabolic

#endif
