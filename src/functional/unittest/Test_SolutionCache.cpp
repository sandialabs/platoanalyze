#include <Teuchos_UnitTestHarness.hpp>

#include "FunctionalInterfaceUtilities.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTestHelpers.hpp"
#include "SolutionCache.hpp"

namespace plato::functional::unittest
{

TEUCHOS_UNIT_TEST(TestSolutionCache, ComputeOnlyWhenDesignChanges)
{
    const std::string tTag = "Length";
    unsigned int tCallCount{0};

    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", 1);

    auto tSolutionCache = SolutionCache(
        [tTag, &tCallCount](const Plato::ScalarVector& aArg)
        {
            Plato::Solutions tSolution;
            tCallCount++;
            return tSolution;
        },

        [&tMesh](const Plato::ScalarVector& aArg) { return hash_current_design(aArg, tMesh); });

    TEST_EQUALITY(tCallCount, 0);
    auto tControl = Plato::ScalarVector("test", 10);

    // initial use "computes solution"
    auto tSolution = tSolutionCache.compute(tControl);
    TEST_EQUALITY(tCallCount, 1);

    // subsequent call to compute with same mesh and same control doesn't call compute function again
    tSolution = tSolutionCache.compute(tControl);
    TEST_EQUALITY(tCallCount, 1);

    // subsequent call to compute with same mesh and new control calls compute function again
    tControl = Plato::ScalarVector("test2", 20);
    tSolution = tSolutionCache.compute(tControl);
    TEST_EQUALITY(tCallCount, 2);

    // subsequent call to compute with new mesh and same control calls compute function again
    tMesh = Plato::TestHelpers::get_box_mesh("TET4", 2);
    tSolution = tSolutionCache.compute(tControl);
    TEST_EQUALITY(tCallCount, 3);

    // one last call to compute with same mesh and same control doesn't call compute function again
    tSolution = tSolutionCache.compute(tControl);
    TEST_EQUALITY(tCallCount, 3);
}
}  // namespace plato::functional::unittest