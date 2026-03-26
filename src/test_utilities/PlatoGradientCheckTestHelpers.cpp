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
}  // namespace Plato::TestHelpers
