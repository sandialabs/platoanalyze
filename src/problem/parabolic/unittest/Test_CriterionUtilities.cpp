#include <Teuchos_LocalTestingHelpers.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include "core_types/PlatoTypes.hpp"
#include "problem/parabolic/CriterionUtilities.hpp"
#include "problem/parabolic/test_utilities/DataUtilities.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::parabolic::unittest
{
TEUCHOS_UNIT_TEST(CriterionUtilities, TrapezoidIntegrationConstant)
{
    constexpr Plato::OrdinalType tNumSteps{86};
    // first time step
    {
        constexpr Plato::OrdinalType tStepIndex{0};
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

TEUCHOS_UNIT_TEST(CriterionUtilities, TrapezoidRuleIntegration)
{
    const std::vector<Plato::Scalar> tValues{0.0, 1.0, 2.0, 3.0, 4.0};
    constexpr Plato::Scalar tTimeStep{1.0};

    constexpr Plato::Scalar tGoldIntegral{8.0};
    TEST_EQUALITY(trapezoidal_rule_integration(tValues, tTimeStep), tGoldIntegral);
}

TEUCHOS_UNIT_TEST(CriterionUtilities, TrapezoidRuleIntegrationVectorOverload)
{
    const std::vector<std::vector<Plato::Scalar>> tVectorValues{
        {0.0, 1.0, 2.0}, {10.0, 11.0, 21.0}, {86.0, 21.0, 20.0}};
    const auto tViewValues = test_utilities::multi_dimension_view_from_vector(tVectorValues);
    constexpr Plato::Scalar tTimeStep{1.0};

    const std::vector<Plato::Scalar> tGoldIntegral{53.0, 22.0, 32.0};  // 53, 22, 32
    const auto tIntegralView = trapezoidal_rule_integration(tViewValues, tTimeStep);
    const auto tIntegralHostView = Plato::TestHelpers::get(tIntegralView);

    TEST_ASSERT(tIntegralHostView.size() == tGoldIntegral.size());
    for (Plato::OrdinalType tIndex = 0; tIndex < tIntegralView.size(); tIndex++)
    {
        TEST_EQUALITY(tIntegralHostView(tIndex), tGoldIntegral[tIndex]);
    }
}
}  // namespace plato::parabolic::unittest
