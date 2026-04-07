#ifndef PLATO_PROBLEM_PARABOLIC_CRITERIONUTILITIES_H
#define PLATO_PROBLEM_PARABOLIC_CRITERIONUTILITIES_H

#include "core_types/PlatoTypes.hpp"

namespace plato::parabolic
{
[[nodiscard]] Plato::Scalar trapezoid_integration_constant(const Plato::OrdinalType aStepIndex,
                                                           const Plato::Scalar aTimeStep,
                                                           const Plato::OrdinalType aNumSteps);
}
#endif
