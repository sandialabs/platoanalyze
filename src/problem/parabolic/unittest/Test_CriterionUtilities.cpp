#include <Teuchos_UnitTestHarness.hpp>

#include "core_types/PlatoTypes.hpp"
#include "problem/parabolic/CriterionUtilities.hpp"

namespace plato::parabolic::unittest
{
TEUCHOS_UNIT_TEST(CriterionUtilities, TrapezoidIntegrationConstant)
{
    constexpr Plato::OrdinalType tNumSteps{86};
    // first time step
    {
        constexpr Plato::OrdinalType tStepIndex{1};
        TEST_EQUALITY(trapezoid_integration_constant(tStepIndex, tNumSteps), 0.5);
    }
    // last time step
    {
        constexpr Plato::OrdinalType tStepIndex{tNumSteps - 1};
        TEST_EQUALITY(trapezoid_integration_constant(tStepIndex, tNumSteps), 0.5);
    }
    // other time step
    {
        constexpr Plato::OrdinalType tStepIndex{21};
        TEST_EQUALITY(trapezoid_integration_constant(tStepIndex, tNumSteps), 1.0);
    }
}
}  // namespace plato::parabolic::unittest
