#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <iomanip>
#include <plato/filter/library/FilterInterface.hpp>
#include <random>

#include "functional/FunctionalInterfaceUtilities.hpp"
#include "functional/HelmholtzFilterInterface.hpp"
#include "functional/unittest/TestMeshSetupTeardown.hpp"

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

    const auto tFilterRadius = 0.900666419935816;
    const auto tFilterParameters = plato::filter::library::FilterParameters{tFilterRadius};
    const auto tFilter = HelmholtzFilterInterface(tFilterParameters);
    // All blocks
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshAllDesignBlocks();
        const auto tResult = tFilter.filter(tAnalysisDomainMesh);
        const auto tRegressionBlockDensities = std::map<int, std::vector<double>>{
            {1, {3.953658063494317, 3.979473301330917, 4.156879952468092, 4.160902857747669, 4.172679917311173}},
            {2,
             {4.156879952468092, 4.160902857747669, 4.172679917311173, 4.353333917626102, 4.390422701843561,
              4.37470229598349}}};

        tTestFunction(tResult, tRegressionBlockDensities);
    }
    // Block 1 fixed
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock1Fixed();
        const auto tResult = tFilter.filter(tAnalysisDomainMesh);
        const auto tRegressionBlockDensities =
            std::map<int, std::vector<double>>{{2,
                                                {4.036530041169163, 4.037929466052052, 4.048074594031434,
                                                 4.243310126679313, 4.281344804443195, 4.265093074165032}}};

        tTestFunction(tResult, tRegressionBlockDensities);
    }
    // Block 2 fixed
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock2Fixed();
        const auto tResult = tFilter.filter(tAnalysisDomainMesh);
        const auto tRegressionBlockDensities = std::map<int, std::vector<double>>{
            {1, {2.566060946362842, 2.593716394303182, 2.596076585940593, 2.610341610583288, 2.621402874904288}}};

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

    const auto tFilterRadius = 0.900666419935816;
    const auto tFilterParameters = plato::filter::library::FilterParameters{tFilterRadius};
    const auto tFilter = HelmholtzFilterInterface(tFilterParameters);
    // All blocks
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshAllDesignBlocks();
        const auto tOnesVector = plato::linear_algebra::DynamicVector(
            std::vector<double>(number_of_analysis_field_variables(tAnalysisDomainMesh), 1.0));
        const auto tResult = tFilter.rowVectorTimesJacobian(tAnalysisDomainMesh, tOnesVector);
        const auto tRegressionBlockDensities =
            std::vector<double>{0.9863222864773449, 0.9872790785970711, 1.000881486065802,  2.31560221871226,
                                0.6646187050426258, 0.3446758279200729, 0.6799924391245759, 1.020627958060224};
        tTestFunction(tResult.stdVector(), tRegressionBlockDensities);
    }
    // Block 1 fixed
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock1Fixed();
        const auto tOnesVector = plato::linear_algebra::DynamicVector(
            std::vector<double>(number_of_analysis_field_variables(tAnalysisDomainMesh), 1.0));
        const auto tResult = tFilter.rowVectorTimesJacobian(tAnalysisDomainMesh, tOnesVector);
        const auto tRegressionBlockDensities =
            std::vector<double>{0.7644493558941634, 1.733194638281614, 0.5023560058242136,
                                0.2720883038682069, 0.533713354495278, 0.8016648300424967};
        tTestFunction(tResult.stdVector(), tRegressionBlockDensities);
    }
    // Block 2 fixed
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock2Fixed();
        const auto tOnesVector = plato::linear_algebra::DynamicVector(
            std::vector<double>(number_of_analysis_field_variables(tAnalysisDomainMesh), 1.0));
        const auto tResult = tFilter.rowVectorTimesJacobian(tAnalysisDomainMesh, tOnesVector);
        const auto tRegressionBlockDensities = std::vector<double>{
            0.6579999011116582, 0.6585681684314523, 0.6199068627397625, 1.468951211216291, 0.4205907211335903};
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
