#include "problem/parabolic/CriterionUtilities.hpp"

#include "core_types/PlatoTypes.hpp"

namespace plato::parabolic
{
Plato::Scalar trapezoid_integration_constant(const Plato::OrdinalType aStepIndex,
                                             const Plato::Scalar aTimeStep,
                                             const Plato::OrdinalType aNumSteps)
{
    return aStepIndex == 1 || aStepIndex == aNumSteps - 1 ? 0.5 * aTimeStep : aTimeStep;
}
}  // namespace plato::parabolic
