#ifndef PLATO_UNITTESTS_UTIL_PLATOGRADIENTCHECKTESTHELPERS
#define PLATO_UNITTESTS_UTIL_PLATOGRADIENTCHECKTESTHELPERS

#include <Teuchos_UnitTestHarness.hpp>
#include <numeric>
#include <plato/utilities/GradientChecker.hpp>
#include <random>
#include <valarray>

#include "core_types/PlatoTypes.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace Plato::TestHelpers
{
/// @brief constructs a gradient checker for a given problem @a aProblem with criterion @a aCriterionName.
/// This is used to check gradient consistency with respect to controls.
template <typename ProblemType>
auto make_criterion_gradient_checker(ProblemType aProblem, const std::string aCriterionName)
    -> plato::utilities::GradientChecker<std::valarray<Plato::Scalar>>
{
    auto tCriterionValue = [&aCriterionName, &aProblem](const std::valarray<Plato::Scalar>& aControlVector)
    {
        const auto tControl = Plato::TestHelpers::create_device_view(aControlVector);
        const auto tStateSolution = aProblem.solution(tControl);
        return aProblem.criterionValue(tControl, tStateSolution, aCriterionName);
    };

    auto tCriterionGradient = [&aCriterionName, &aProblem](const std::valarray<Plato::Scalar>& aControlVector,
                                                           const std::valarray<Plato::Scalar>& aDirection)
    {
        const auto tControl = Plato::TestHelpers::create_device_view(aControlVector);
        const auto tStateSolution = aProblem.solution(tControl);
        const auto tGradient = aProblem.criterionGradient(tControl, tStateSolution, aCriterionName);
        const auto tHostGradient = Plato::TestHelpers::get(tGradient);
        return std::inner_product(Kokkos::Experimental::begin(tHostGradient), Kokkos::Experimental::end(tHostGradient),
                                  begin(aDirection), 0.0);
    };

    return plato::utilities::GradientChecker<std::valarray<Plato::Scalar>>{tCriterionValue, tCriterionGradient};
}

/// @brief uses the gradient checker @a aGradientChecker to perform a gradient check about the control values @a aX.
/// The gradient check paramters are defined by input @a aGradientCheckParameters and first order truncation error is
/// checked against @a aTruncationErrorTolerance.
void check_control_gradient(const plato::utilities::GradientChecker<std::valarray<Plato::Scalar>>& aGradientChecker,
                            const plato::utilities::GradientCheckParameters& aGradientCheckParameters,
                            const std::valarray<Plato::Scalar>& aX,
                            const Plato::Scalar aTruncationErrorTolerance,
                            Teuchos::FancyOStream& aOutStream,
                            bool& aSuccess);

namespace detail
{
using RandomEngineSeedType = std::default_random_engine::result_type;
/// @brief Create a valarray of size @a aSize whose components are random values between [-1,1] generated with a
/// std::default_random_engine using seed @a aSeed
[[nodiscard]] auto random_perturbation(const Plato::OrdinalType aSize, const RandomEngineSeedType aSeed)
    -> std::valarray<double>;
}  // namespace detail

}  // namespace Plato::TestHelpers

#endif
