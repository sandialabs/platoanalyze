#ifndef PLATO_PROBLEM_PARABOLIC_CRITERIONUTILITIES_H
#define PLATO_PROBLEM_PARABOLIC_CRITERIONUTILITIES_H

#include <vector>

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace plato::parabolic
{
static constexpr Plato::OrdinalType kFirstTimeStep{0};

/// @a brief, provides trapezoidal rule integration constant (i.e. 0.5 for steps 0 and N and 1.0 for others)
[[nodiscard]] Plato::Scalar trapezoid_integration_constant(const Plato::OrdinalType aStepIndex,
                                                           const Plato::OrdinalType aNumSteps);

/// @a brief Performs trapezoidal rule time integration over values in @a aTimeStepValues with a constant time step @a
/// aTimeStep
[[nodiscard]] Plato::Scalar trapezoidal_rule_integration(const std::vector<Plato::Scalar>& aTimeStepValues,
                                                         const Plato::Scalar aTimeStep);

/// @a brief overload to perform trapezoidal rule time integration over vector quantities. Needed for criterion
/// gradients.
[[nodiscard]] Plato::ScalarVector trapezoidal_rule_integration(const Plato::ScalarMultiVector aTimeStepValues,
                                                               const Plato::Scalar aTimeStep);
}  // namespace plato::parabolic
#endif
