#ifndef PLATO_UNITTESTS_UTIL_PLATOGRADIENTCHECKTESTHELPERS
#define PLATO_UNITTESTS_UTIL_PLATOGRADIENTCHECKTESTHELPERS

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <numeric>
#include <plato/test_utilities/GradientChecker.hpp>
#include <random>
#include <valarray>

#include "core_types/PlatoTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace Plato::TestHelpers
{
/// @brief Create a valarray of size @a aSize whose components are random values between [-1,1] generated with @a
/// aRandomEngine. The vector is then normalized.
auto random_perturbation_valarray(const Plato::OrdinalType aSize, std::default_random_engine& aRandomEngine)
    -> std::valarray<double>;

/// @brief perform a gradient check for problem defined by @a aParamList over the mesh @a aMesh for the criterion
/// with name @a aCriterionName.
/// The gradient check paramters are defined by input @a aGradientCheckParameters.
///
/// @tparam CreateProblem callable for constructing an instance of the problem object.
///         Must have the following signature:
///         Plato::AbstractProblem(const Plato::Mesh& aMesh,
///         Teuchos::ParameterList& aParameterList)
template <typename CreateProblem>
void check_gradient_over_mesh(Teuchos::ParameterList& aParamList,
                              const CreateProblem& aCreateProblem,
                              const std::string aCriterionName,
                              const Plato::Mesh& aMesh,
                              const Plato::Scalar aControlValue,
                              const plato::test_utilities::GradientCheckParameters& aGradientCheckParameters,
                              const Plato::Scalar aTruncationErrorTolerance,
                              Teuchos::FancyOStream& aOutStream,
                              bool& aSuccess)
{
    auto tProblem = aCreateProblem(aMesh, aParamList);

    auto tCriterionValue = [&aCriterionName, &tProblem](const std::valarray<Plato::Scalar>& aControlVector)
    {
        const auto tControl = Plato::TestHelpers::create_device_view(aControlVector);
        const auto tStateSolution = tProblem.solution(tControl);
        return tProblem.criterionValue(tControl, tStateSolution, aCriterionName);
    };

    auto tCriterionGradient = [&aCriterionName, &tProblem](const std::valarray<Plato::Scalar>& aControlVector,
                                                           const std::valarray<Plato::Scalar>& aDirection)
    {
        const auto tControl = Plato::TestHelpers::create_device_view(aControlVector);
        const auto tStateSolution = tProblem.solution(tControl);
        const auto tGradient = tProblem.criterionGradient(tControl, tStateSolution, aCriterionName);
        const auto tHostGradient = Plato::TestHelpers::get(tGradient);
        return std::inner_product(Kokkos::Experimental::begin(tHostGradient), Kokkos::Experimental::end(tHostGradient),
                                  begin(aDirection), 0.0);
    };

    const plato::test_utilities::GradientChecker<std::valarray<Plato::Scalar>> tGradientChecker{tCriterionValue,
                                                                                                tCriterionGradient};

    const auto tNumNodes = aMesh->NumNodes();
    const std::valarray<Plato::Scalar> tControl(aControlValue, tNumNodes);

    auto tRandomEngine = std::default_random_engine{123};
    const auto tPerturbationDirection = random_perturbation_valarray(tNumNodes, tRandomEngine);

    const auto tMaxTruncationError =
        tGradientChecker.maxFirstOrderTruncationError(tControl, tPerturbationDirection, aGradientCheckParameters);
    TEUCHOS_TEST_ASSERT(tMaxTruncationError < aTruncationErrorTolerance, aOutStream, aSuccess);
    if (!aSuccess)
    {
        std::cout << "\n Failing gradient check table is: \n";
        std::cout << tGradientChecker.table(tControl, tPerturbationDirection, aGradientCheckParameters);
        std::cout << "\n Max first order truncation error is: " << tMaxTruncationError << "\n";
    }
}
}  // namespace Plato::TestHelpers

#endif
