
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
