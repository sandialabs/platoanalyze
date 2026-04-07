#include "test_utilities/PlatoGradientCheckTestHelpers.hpp"

#include <algorithm>
#include <random>
#include <valarray>

#include "core_types/PlatoTypes.hpp"

namespace Plato::TestHelpers
{
void check_control_gradient(
    const plato::test_utilities::GradientChecker<std::valarray<Plato::Scalar>>& aGradientChecker,
    const plato::test_utilities::GradientCheckParameters& aGradientCheckParameters,
    const std::valarray<Plato::Scalar>& aX,
    const Plato::Scalar aTruncationErrorTolerance,
    Teuchos::FancyOStream& aOutStream,
    bool& aSuccess)
{
    const detail::RandomEngineSeedType tSeed{123};
    const auto tPerturbationDirection = detail::random_perturbation(aX.size(), tSeed);

    const auto tMaxTruncationError =
        aGradientChecker.maxFirstOrderTruncationError(aX, tPerturbationDirection, aGradientCheckParameters);
    TEUCHOS_TEST_ASSERT(tMaxTruncationError < aTruncationErrorTolerance, aOutStream, aSuccess);
    if (!aSuccess)
    {
        aOutStream << "\n Failing gradient check table is: \n";
        aOutStream << aGradientChecker.table(aX, tPerturbationDirection, aGradientCheckParameters);
        aOutStream << "\n Max first order truncation error is: " << tMaxTruncationError << "\n";
    }
}

namespace detail
{
auto random_perturbation(const Plato::OrdinalType aSize, const RandomEngineSeedType aSeed) -> std::valarray<double>
{
    assert(aSize != 0);
    std::uniform_real_distribution<double> tDistribution{-1, 1};

    auto tRandomEngine = std::default_random_engine{aSeed};

    auto tRandomValues = std::valarray<double>(aSize);
    std::ranges::generate_n(begin(tRandomValues), static_cast<Plato::OrdinalType>(aSize),
                            [&tRandomEngine, &tDistribution]() { return tDistribution(tRandomEngine); });

    return tRandomValues;
}

}  // namespace detail
}  // namespace Plato::TestHelpers
