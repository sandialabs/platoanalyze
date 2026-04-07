#include <Teuchos_UnitTestHarness.hpp>
#include <algorithm>

#include "core_types/PlatoTypes.hpp"
#include "test_utilities/PlatoGradientCheckTestHelpers.hpp"

namespace Plato::TestHelpers
{
TEUCHOS_UNIT_TEST(PlatoTestHelpers, DetailRandomPerturbation)
{
    constexpr Plato::OrdinalType tSize{4};
    const detail::RandomEngineSeedType tSeed{123};
    const auto tPerturbationDirection = detail::random_perturbation(tSize, tSeed);

    TEST_ASSERT(tPerturbationDirection.size() == tSize);

    const auto tMin = *std::min_element(begin(tPerturbationDirection), end(tPerturbationDirection));
    const auto tMax = *std::max_element(begin(tPerturbationDirection), end(tPerturbationDirection));

    TEST_ASSERT(tMax <= 1.0);
    TEST_ASSERT(tMin >= -1.0);
}
}  // namespace Plato::TestHelpers
