#include "problem/parabolic/CriterionUtilities.hpp"

#include <vector>

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace plato::parabolic
{
Plato::Scalar trapezoid_integration_constant(const Plato::OrdinalType aStepIndex, const Plato::OrdinalType aNumSteps)
{
    return aStepIndex == kFirstTimeStep || aStepIndex == aNumSteps - 1 ? 0.5 : 1.0;
}

Plato::Scalar trapezoidal_rule_integration(const std::vector<Plato::Scalar>& aTimeStepValues,
                                           const Plato::Scalar aTimeStep)
{
    const auto tNumSteps = aTimeStepValues.size();
    Plato::Scalar tIntegral{0.0};
    for (Plato::OrdinalType tStepIndex = kFirstTimeStep; tStepIndex < tNumSteps; ++tStepIndex)
    {
        tIntegral += trapezoid_integration_constant(tStepIndex, tNumSteps) * aTimeStep * aTimeStepValues[tStepIndex];
    }
    return tIntegral;
}

Plato::ScalarVector trapezoidal_rule_integration(const Plato::ScalarMultiVector aTimeStepValues,
                                                 const Plato::Scalar aTimeStep)
{
    const auto tNumSteps = aTimeStepValues.extent(0);
    Plato::ScalarVector tIntegral("integrated vector quantity", aTimeStepValues.extent(1));
    for (Plato::OrdinalType tStepIndex = kFirstTimeStep; tStepIndex < tNumSteps; ++tStepIndex)
    {
        Plato::ScalarVector tValues = Kokkos::subview(aTimeStepValues, tStepIndex, Kokkos::ALL());
        Plato::blas1::scale(trapezoid_integration_constant(tStepIndex, tNumSteps) * aTimeStep, tValues);
        Plato::blas1::axpy(1.0, tValues, tIntegral);
    }
    return tIntegral;
}
}  // namespace plato::parabolic
