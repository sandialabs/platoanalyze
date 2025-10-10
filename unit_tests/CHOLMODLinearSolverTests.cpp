#include <Kokkos_StdAlgorithms.hpp>

#include "Teuchos_UnitTestHarness.hpp"
#include "alg/CHOLMODLinearSolver.hpp"
#include "util/PlatoMathTestHelpers.hpp"

namespace
{
constexpr auto kTolerance = 1e-14;
constexpr auto kNumberOfRows = 4U;
const auto kTriDiagonalRowMap4x4 = std::vector<Plato::OrdinalType>{0, 2, 5, 8, 10};
const auto kTriDiagonalColMap4x4 = std::vector<Plato::OrdinalType>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3};

/// @brief Returns a 4x4 symmetric positive definite matrix.
///
/// The entries are:
/// @code
///    A = [ 2    -1     0     0
///         -1     2    -1     0
///          0    -1     2    -1
///          0     0    -1     2];
/// @endcode
[[nodiscard]] auto symmetric_positive_definite_matrix()
{
    const auto tValuesA = std::vector<Plato::Scalar>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
    return Plato::TestHelpers::square_crs_matrix(kNumberOfRows, kTriDiagonalRowMap4x4, kTriDiagonalColMap4x4, tValuesA);
}

/// @brief Returns a 4x4 symmetric negative definite matrix.
///
/// The entries are:
/// @code
///    A = [-2    -1     0     0
///         -1    -2    -1     0
///          0    -1    -2    -1
///          0     0    -1    -2];
/// @endcode
[[nodiscard]] auto symmetric_negative_definite_matrix()
{
    const auto tNumberOfRows = 4U;
    const auto tValuesA = std::vector<Plato::Scalar>{-2.0, -1.0, -1.0, -2.0, -1.0, -1.0, -2.0, -1.0, -1.0, -2.0};
    return Plato::TestHelpers::square_crs_matrix(tNumberOfRows, kTriDiagonalRowMap4x4, kTriDiagonalColMap4x4, tValuesA);
}

/// @brief Returns a 4x4 symmetric negative definite matrix.
///
/// The entries are:
/// @code
///    A = [ 2    -1     0     0
///         -1     2    -1     0
///          0    -1    -2    -1
///          0     0    -1    -2];
/// @endcode
[[nodiscard]] auto symmetric_indefinite_matrix()
{
    const auto tNumberOfRows = 4U;
    const auto tValuesA = std::vector<Plato::Scalar>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, -2.0, -1.0, -1.0, -2.0};
    return Plato::TestHelpers::square_crs_matrix(tNumberOfRows, kTriDiagonalRowMap4x4, kTriDiagonalColMap4x4, tValuesA);
}

/// @brief Returns a vector usable as a right-hand-side for solving systems with the above matrices.
///
/// The entries are:
/// @code
///    b = [0 1 1 0]';
/// @endcode
[[nodiscard]] auto rhs()
{
    const auto tRHSVector = std::vector{0.0, 1.0, 1.0, 0.0};
    auto tRHSView = Plato::ScalarVector{"RHS", tRHSVector.size()};
    Plato::TestHelpers::set_view_from_vector(tRHSView, tRHSVector);
    return tRHSView;
}

void solve_and_check_solution(const Plato::CrsMatrixType aMatrix,
                              const Plato::ScalarVector aRHS,
                              const Plato::LinearSystemType aSystemType,
                              const std::vector<double>& aExpectedSolution,
                              Teuchos::FancyOStream& aOutStream,
                              bool& aSuccess)
{
    const auto tSolutionView = Plato::ScalarVector{"Solution", aRHS.size()};
    auto tCholmodSolver = Plato::alg::CHOLMODLinearSolver{Teuchos::ParameterList{}, aSystemType};
    tCholmodSolver.innerSolve(aMatrix, tSolutionView, aRHS);

    const auto tTestResult = Plato::TestHelpers::is_near(tSolutionView, aExpectedSolution, kTolerance);
    TEUCHOS_TEST_ASSERT(tTestResult.first, aOutStream, aSuccess);
    aOutStream << tTestResult.second;
}

}  // namespace

TEUCHOS_UNIT_TEST(CHOLMODSolver, ConvertCSRtoCHOLMODSparse)
{
    auto tCholmodCommon = Plato::alg::CholmodCommonSetupTeardown{Plato::LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE};
    const auto tCRSMatrix = Plato::alg::constructCSRMatrix(symmetric_positive_definite_matrix());
    const auto* const tCholmodSparse =
        Plato::alg::convertSymmetricCSRtoCHOLMODSparse(tCRSMatrix, &tCholmodCommon.mValue);

    TEST_INEQUALITY_CONST(tCholmodSparse, nullptr);
    TEST_EQUALITY(tCholmodSparse->nrow, kNumberOfRows);
    TEST_EQUALITY(tCholmodSparse->ncol, kNumberOfRows);

    constexpr auto tExpectedNumberOfNonZero = 7U;
    const auto tExpectedEntries = std::vector{2.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0};
    for (auto tIndex = 0U; tIndex < tExpectedNumberOfNonZero; ++tIndex)
    {
        TEST_EQUALITY(tExpectedEntries.at(tIndex), static_cast<double*>(tCholmodSparse->x)[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(CHOLMODSolver, CholmodCommonSetupTeardown)
{
    {
        const auto tCholmodCommon =
            Plato::alg::CholmodCommonSetupTeardown{Plato::LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE};
        TEST_EQUALITY(tCholmodCommon.mValue.supernodal, CHOLMOD_AUTO);
    }
    {
        const auto tCholmodCommon =
            Plato::alg::CholmodCommonSetupTeardown{Plato::LinearSystemType::SYMMETRIC_INDEFINITE};
        TEST_EQUALITY(tCholmodCommon.mValue.supernodal, CHOLMOD_SIMPLICIAL);
    }
}

TEUCHOS_UNIT_TEST(CHOLMODSolver, CholmodFactorSetupTeardown)
{
    auto tCholmodCommon = Plato::alg::CholmodCommonSetupTeardown{Plato::LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE};
    auto tMatrix = symmetric_positive_definite_matrix();
    const auto tCRSMatrix = Plato::alg::constructCSRMatrix(tMatrix);

    // Move ctor
    {
        auto tCholmodFactor1 = Plato::alg::CholmodFactorSetupTeardown{
            convertSymmetricCSRtoCHOLMODSparse(tCRSMatrix, &tCholmodCommon.mValue), std::ref(tCholmodCommon)};

        const auto* const tCholmodFactorPtr = tCholmodFactor1.mValue;
        const auto tCholmodFactor2 = std::move(tCholmodFactor1);

        TEST_EQUALITY_CONST(tCholmodFactor1.mValue, nullptr);
        TEST_EQUALITY(tCholmodFactor2.mValue, tCholmodFactorPtr);
    }
    // Move assignment
    {
        auto tCholmodFactor1 = Plato::alg::CholmodFactorSetupTeardown{
            convertSymmetricCSRtoCHOLMODSparse(tCRSMatrix, &tCholmodCommon.mValue), std::ref(tCholmodCommon)};
        const auto* const tCholmodFactor1Ptr = tCholmodFactor1.mValue;

        auto tCholmodFactor2 = Plato::alg::CholmodFactorSetupTeardown{
            convertSymmetricCSRtoCHOLMODSparse(tCRSMatrix, &tCholmodCommon.mValue), std::ref(tCholmodCommon)};

        tCholmodFactor2 = std::move(tCholmodFactor1);

        TEST_EQUALITY_CONST(tCholmodFactor1.mValue, nullptr);
        TEST_EQUALITY(tCholmodFactor2.mValue, tCholmodFactor1Ptr);
    }
}

TEUCHOS_UNIT_TEST(CHOLMODSolver, SymmetricPositiveDefinite)
{
    const auto tExpected = std::vector{1.0, 2.0, 2.0, 1.0};
    solve_and_check_solution(symmetric_positive_definite_matrix(), rhs(),
                             Plato::LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE, tExpected, out, success);
}

TEUCHOS_UNIT_TEST(CHOLMODSolver, SymmetricNegativeDefinite)
{
    const auto tExpected = std::vector{0.2, -0.4, -0.4, 0.2};
    solve_and_check_solution(symmetric_negative_definite_matrix(), rhs(), Plato::LinearSystemType::SYMMETRIC_INDEFINITE,
                             tExpected, out, success);
}

TEUCHOS_UNIT_TEST(CHOLMODSolver, SymmetricIndefinite)
{
    const auto tExpected =
        std::vector{7.692307692307694e-02, 1.538461538461539e-01, -7.692307692307692e-01, 3.846153846153846e-01};
    solve_and_check_solution(symmetric_indefinite_matrix(), rhs(), Plato::LinearSystemType::SYMMETRIC_INDEFINITE,
                             tExpected, out, success);
}

TEUCHOS_UNIT_TEST(CHOLMODSolver, TwoSolvesDifferentMatricesSameSparsityPattern)
{
    auto tMatrix = symmetric_positive_definite_matrix();

    const auto tSolutionView = Plato::ScalarVector{"Solution", static_cast<unsigned>(tMatrix.numRows())};
    auto tCholmodSolver =
        Plato::alg::CHOLMODLinearSolver{Teuchos::ParameterList{}, Plato::LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE};

    const auto tExpected = std::vector{1.0, 2.0, 2.0, 1.0};
    {
        tCholmodSolver.innerSolve(tMatrix, tSolutionView, rhs());
        const auto tTestResult = Plato::TestHelpers::is_near(tSolutionView, tExpected, kTolerance);
        TEST_ASSERT(tTestResult.first);
        out << tTestResult.second;
    }
    {
        constexpr auto tMultiplier = 4.0;
        auto tMatrixEntries = tMatrix.entries();
        std::transform(Kokkos::Experimental::begin(tMatrixEntries), Kokkos::Experimental::end(tMatrixEntries),
                       Kokkos::Experimental::begin(tMatrixEntries),
                       [](const double tValue) { return tMultiplier * tValue; });
        tMatrix.setEntries(tMatrixEntries);

        tCholmodSolver.innerSolve(tMatrix, tSolutionView, rhs());
        auto tExpectedWithMultiplier = tExpected;
        std::transform(tExpectedWithMultiplier.begin(), tExpectedWithMultiplier.end(), tExpectedWithMultiplier.begin(),
                       [](const double tValue) { return tValue / tMultiplier; });

        const auto tTestResult = Plato::TestHelpers::is_near(tSolutionView, tExpectedWithMultiplier, kTolerance);
        TEST_ASSERT(tTestResult.first);
        out << tTestResult.second;
    }
}

TEUCHOS_UNIT_TEST(CHOLMODSolver, NonSymmetricMatrix)
{
    const auto kTriDiagonalColMap4x4 = std::vector<Plato::OrdinalType>{0, 1, 0, 1, 3, 1, 2, 3, 2, 3};
    const auto tValuesA = std::vector<Plato::Scalar>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
    const auto tNonsymmetricMatrix =
        Plato::TestHelpers::square_crs_matrix(kNumberOfRows, kTriDiagonalRowMap4x4, kTriDiagonalColMap4x4, tValuesA);

    const auto tSolutionView = Plato::ScalarVector{"Solution", static_cast<unsigned>(tNonsymmetricMatrix.numRows())};
    auto tCholmodSolver =
        Plato::alg::CHOLMODLinearSolver{Teuchos::ParameterList{}, Plato::LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE};

    TEST_THROW(tCholmodSolver.innerSolve(tNonsymmetricMatrix, tSolutionView, rhs()), std::runtime_error);
}
