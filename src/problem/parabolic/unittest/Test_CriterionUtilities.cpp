#include <Teuchos_UnitTestHarness.hpp>

#include "core_types/PlatoTypes.hpp"
#include "problem/parabolic/CriterionUtilities.hpp"

namespace plato::parabolic::unittest
{
TEUCHOS_UNIT_TEST(CriterionUtilities, TrapezoidIntegrationConstant)
{
    constexpr Plato::OrdinalType tNumSteps{86};
    constexpr Plato::Scalar tTimeStep{21.0};
    // first time step
    {
        constexpr Plato::OrdinalType tStepIndex{1};
        TEST_EQUALITY(trapezoid_integration_constant(tStepIndex, tTimeStep, tNumSteps), 0.5 * tTimeStep);
    }
    // last time step
    {
        constexpr Plato::OrdinalType tStepIndex{tNumSteps - 1};
        TEST_EQUALITY(trapezoid_integration_constant(tStepIndex, tTimeStep, tNumSteps), 0.5 * tTimeStep);
    }
    // other time step
    {
        constexpr Plato::OrdinalType tStepIndex{21};
        TEST_EQUALITY(trapezoid_integration_constant(tStepIndex, tTimeStep, tNumSteps), tTimeStep);
    }
}
}  // namespace plato::parabolic::unittest
