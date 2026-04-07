#include "test_utilities/PlatoGradientCheckTestHelpers.hpp"

#include <algorithm>
#include <random>
#include <valarray>

#include "core_types/PlatoTypes.hpp"

namespace Plato::TestHelpers
{
auto random_perturbation_valarray(const Plato::OrdinalType aSize, std::default_random_engine& aRandomEngine)
    -> std::valarray<double>
{
    assert(aSize != 0);
    std::uniform_real_distribution<double> tDistribution{-1, 1};

    auto tRandomValues = std::valarray<double>(aSize);
    std::ranges::generate_n(begin(tRandomValues), static_cast<Plato::OrdinalType>(aSize),
                            [&aRandomEngine, &tDistribution]() { return tDistribution(aRandomEngine); });

    const auto tNorm =
        std::sqrt(std::inner_product(begin(tRandomValues), end(tRandomValues), begin(tRandomValues), 0.0));
    tRandomValues /= tNorm;

    return tRandomValues;
}

void check_control_gradient(
    const plato::test_utilities::GradientChecker<std::valarray<Plato::Scalar>>& aGradientChecker,
    const plato::test_utilities::GradientCheckParameters& aGradientCheckParameters,
    const std::valarray<Plato::Scalar>& aX,
    const Plato::Scalar aTruncationErrorTolerance,
    Teuchos::FancyOStream& aOutStream,
    bool& aSuccess)
{
    auto tRandomEngine = std::default_random_engine{123};
    const auto tPerturbationDirection = random_perturbation_valarray(aX.size(), tRandomEngine);

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
}  // namespace Plato::TestHelpers
