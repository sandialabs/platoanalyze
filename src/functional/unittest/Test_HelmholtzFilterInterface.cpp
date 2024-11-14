#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <iomanip>
#include <plato/filter/FilterInterface.hpp>
#include <random>

#include "FunctionalInterfaceUtilities.hpp"
#include "HelmholtzFilterInterface.hpp"
#include "TestMeshSetupTeardown.hpp"

namespace plato::functional::unittest
{

namespace
{
template <typename Engine>
auto random_vector(const std::size_t aSize, Engine& aRandomEngine) -> linear_algebra::DynamicVector<double>
{
    auto tVector = std::vector<double>(aSize);
    constexpr auto tMean = 0.0;
    constexpr auto tStandardDeviation = 1.0;
    std::generate(tVector.begin(), tVector.end(),
                  [&aRandomEngine, tDistribution = std::normal_distribution<double>{
                                       tMean, tStandardDeviation}]() mutable { return tDistribution(aRandomEngine); });
    return linear_algebra::DynamicVector<double>(std::move(tVector));
}

template <typename F>
auto weighted_inner_product(const plato::linear_algebra::DynamicVector<double>& aVectorLeft,
                            const F& aRowVectorMatrixProduct,
                            const plato::linear_algebra::DynamicVector<double>& aVectorRight,
                            const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh)
{
    const auto tLeftVectorTimesMatrix = aRowVectorMatrixProduct(aAnalysisDomainMesh, aVectorLeft);
    return tLeftVectorTimesMatrix.dot(aVectorRight);
}

}  // namespace

TEUCHOS_UNIT_TEST(HelmholtzFilterInterface, FilterRegression)
{
    const auto tTestFixture = TestMeshSetupTeardown{};

    const auto tTestFunction = [&](const plato::analysis::AnalysisDomainMesh& aResult,
                                   const std::map<int, std::vector<double>>& aRegressionBlockDensities)
    {
        TEST_EQUALITY(aResult.mBlockScalarField.size(), aRegressionBlockDensities.size());
        for (const auto& [tBlockID, tResultDensities] : aResult.mBlockScalarField)
        {
            const auto& tRegressionDensities = aRegressionBlockDensities.at(tBlockID);
            TEST_EQUALITY(tRegressionDensities.size(), tResultDensities.size());
            constexpr auto tTolerance = 1e-14;
            for (auto tIndex = 0U; tIndex < tRegressionDensities.size(); ++tIndex)
            {
                TEST_FLOATING_EQUALITY(tRegressionDensities[tIndex], tResultDensities[tIndex].mValue, tTolerance);
            }
        }
    };

    const auto tFilterRadius = 0.26;
    const auto tFilterParameters = plato::filter::library::FilterParameters{tFilterRadius};
    const auto tFilter = HelmholtzFilterInterface(tFilterParameters);
    // All blocks
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshAllDesignBlocks();
        const auto tResult = tFilter.filter(tAnalysisDomainMesh);
        const auto tRegressionBlockDensities = std::map<int, std::vector<double>>{
            {1, {3.989534972684765, 4.005611590018745, 4.167508297994873, 4.163367110053167, 4.171356751583184}},
            {2,
             {4.167508297994873, 4.163367110053167, 4.171356751583184, 4.322358904846153, 4.350306252767223,
              4.334260244661879}}};

        tTestFunction(tResult, tRegressionBlockDensities);
    }
    // Block 1 fixed
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock1Fixed();
        const auto tResult = tFilter.filter(tAnalysisDomainMesh);
        const auto tRegressionBlockDensities =
            std::map<int, std::vector<double>>{{2,
                                                {4.045611765166145, 4.038671230784396, 4.045563528601696,
                                                 4.211065651063589, 4.240095189763867, 4.223408884391668}}};

        tTestFunction(tResult, tRegressionBlockDensities);
    }
    // Block 2 fixed
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock2Fixed();
        const auto tResult = tFilter.filter(tAnalysisDomainMesh);
        const auto tRegressionBlockDensities = std::map<int, std::vector<double>>{
            {1, {2.58498312091394, 2.604892249666857, 2.581443556225109, 2.592666672332953, 2.605060695617186}}};

        tTestFunction(tResult, tRegressionBlockDensities);
    }
}

TEUCHOS_UNIT_TEST(HelmholtzFilterInterface, JacobianRegression)
{
    const auto tTestFixture = TestMeshSetupTeardown{};

    const auto tTestFunction =
        [&](const std::vector<double>& aResultJacobian, const std::vector<double>& aRegressionJacobian)
    {
        TEST_EQUALITY(aResultJacobian.size(), aRegressionJacobian.size());
        for (auto tIndex = 0U; tIndex < aRegressionJacobian.size(); ++tIndex)
        {
            constexpr auto tTolerance = 1e-14;
            TEST_FLOATING_EQUALITY(aResultJacobian.at(tIndex), aRegressionJacobian.at(tIndex), tTolerance);
        }
    };

    const auto tFilterRadius = 0.26;
    const auto tFilterParameters = plato::filter::library::FilterParameters{tFilterRadius};
    const auto tFilter = HelmholtzFilterInterface(tFilterParameters);
    // All blocks
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshAllDesignBlocks();
        const auto tOnesVector = plato::linear_algebra::DynamicVector(
            std::vector<double>(number_of_analysis_field_variables(tAnalysisDomainMesh), 1.0));
        const auto tResult = tFilter.rowVectorTimesJacobian(tAnalysisDomainMesh, tOnesVector);
        const auto tRegressionBlockDensities =
            std::vector<double>{0.9857300023534444, 0.9866024254916289, 1.004803010671341,  2.324358568896947,
                                0.6648815169928183, 0.342308097769717,  0.6752612305036949, 1.016055147320401};
        tTestFunction(tResult.stdVector(), tRegressionBlockDensities);
    }
    // Block 1 fixed
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock1Fixed();
        const auto tOnesVector = plato::linear_algebra::DynamicVector(
            std::vector<double>(number_of_analysis_field_variables(tAnalysisDomainMesh), 1.0));
        const auto tResult = tFilter.rowVectorTimesJacobian(tAnalysisDomainMesh, tOnesVector);
        const auto tRegressionBlockDensities =
            std::vector<double>{0.7683236710634752, 1.730773305817584,  0.5013065698002031,
                                0.2692266941824142, 0.5269350168014131, 0.7946398770386152};
        tTestFunction(tResult.stdVector(), tRegressionBlockDensities);
    }
    // Block 2 fixed
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock2Fixed();
        const auto tOnesVector = plato::linear_algebra::DynamicVector(
            std::vector<double>(number_of_analysis_field_variables(tAnalysisDomainMesh), 1.0));
        const auto tResult = tFilter.rowVectorTimesJacobian(tAnalysisDomainMesh, tOnesVector);
        const auto tRegressionBlockDensities = std::vector<double>{
            0.6537499612395944, 0.6542467484355006, 0.6146358731048033, 1.47020614276741, 0.4187273429521783};
        tTestFunction(tResult.stdVector(), tRegressionBlockDensities);
    }
}

TEUCHOS_UNIT_TEST(HelmholtzFilterInterface, AdjointJacobianWeightedInnerProduct)
{
    const auto tTestFixture = TestMeshSetupTeardown{};

    const auto tFilterRadius = 1.1;
    const auto tFilterParameters = plato::filter::library::FilterParameters{tFilterRadius};
    const auto tFilter = HelmholtzFilterInterface{tFilterParameters};
    const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshAllDesignBlocks();
    const auto tNumberOfAnalysisVariables = number_of_analysis_field_variables(tAnalysisDomainMesh);

    const auto tJacobian = [&tFilter](const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                      const plato::linear_algebra::DynamicVector<double>& aVector)
    { return tFilter.rowVectorTimesJacobian(aAnalysisDomainMesh, aVector); };
    const auto tAdjointJacobian = [&tFilter](const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                             const plato::linear_algebra::DynamicVector<double>& aVector)
    { return tFilter.rowVectorTimesAdjointJacobian(aAnalysisDomainMesh, aVector); };
    constexpr auto tNumberOfTestVectors = 50U;
    auto tRandomEngine = std::default_random_engine{};
    for (auto tCount = 0U; tCount < tNumberOfAnalysisVariables; ++tCount)
    {
        const auto tX = random_vector(tNumberOfAnalysisVariables, tRandomEngine);
        const auto tY = random_vector(tNumberOfAnalysisVariables, tRandomEngine);
        const auto tInnerProduct = weighted_inner_product(tX, tJacobian, tY, tAnalysisDomainMesh);
        const auto tAdjointInnerProduct = weighted_inner_product(tY, tAdjointJacobian, tX, tAnalysisDomainMesh);

        constexpr auto tTolerance = 1e-14;
        TEST_FLOATING_EQUALITY(tInnerProduct, tAdjointInnerProduct, tTolerance);
    }
}

}  // namespace plato::functional::unittest
