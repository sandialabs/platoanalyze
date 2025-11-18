#include <Teuchos_UnitTestHarness.hpp>

#include "solver/umfpack/UMFPACKLinearSolver.hpp"
#include "test_utilities/PlatoMathTestHelpers.hpp"

namespace
{
constexpr auto kTolerance = 1e-14;
constexpr auto kNumberOfRows = 4U;

/// @brief Returns a 4x4 sparse, asymmetric
///
/// The entries are:
/// @code
///    A = [ 2     0     0     8
///          1     2     1     0
///          0    -1    -2     1
///          0     4     1    -2];
/// @endcode
[[nodiscard]] auto asymmetric_matrix()
{
    constexpr auto kTolerance = 1e-14;
    const auto kTriDiagonalRowMap4x4 = std::vector<Plato::OrdinalType>{0, 2, 5, 8, 11};
    const auto kTriDiagonalColMap4x4 = std::vector<Plato::OrdinalType>{0, 3, 0, 1, 2, 1, 2, 3, 1, 2, 3};
    const auto tValuesA = std::vector<Plato::Scalar>{2.0, 8.0, 1.0, 2.0, 1.0, -1.0, -2.0, 1.0, 4.0, 1.0, -2.0};
    return Plato::TestHelpers::square_crs_matrix(kNumberOfRows, kTriDiagonalRowMap4x4, kTriDiagonalColMap4x4, tValuesA);
}

[[nodiscard]] auto rhs() -> Plato::ScalarVector
{
    const auto tRHSVector = std::vector{1.0, 2.0, 4.0, 8.0};
    auto tRHSView = Plato::ScalarVector{"RHS", tRHSVector.size()};
    Plato::TestHelpers::set_view_from_vector(tRHSView, tRHSVector);
    return tRHSView;
}

[[nodiscard]] auto solve_and_check_solution(const Plato::CrsMatrixType aMatrix,
                                            const std::vector<double>& aExpectedSolution)
{
    const auto tRHS = rhs();
    const auto tSolutionView = Plato::ScalarVector{"Solution", tRHS.size()};
    auto tSolver = Plato::alg::UMFPACKLinearSolver{Teuchos::ParameterList{}};
    tSolver.innerSolve(aMatrix, tSolutionView, tRHS);
    return Plato::TestHelpers::is_near(tSolutionView, aExpectedSolution, kTolerance);
}

}  // namespace

TEUCHOS_UNIT_TEST(UMFPACKSolver, SparseMatrixSolution)
{
    const auto tMatrix = asymmetric_matrix();
    const auto tExpected = std::vector{-0.6, 2.975, -3.35, 0.275};
    const auto tResult = solve_and_check_solution(tMatrix, tExpected);

    TEST_ASSERT(tResult.first);
    out << tResult.second;
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, MultipleSolves)
{
    auto tMatrix = asymmetric_matrix();
    // Solve
    {
        const auto tExpected = std::vector{-0.6, 2.975, -3.35, 0.275};
        const auto tResult = solve_and_check_solution(tMatrix, tExpected);

        TEST_ASSERT(tResult.first);
        out << tResult.second;
    }
    // Change entries, but not sparsity pattern
    {
        // Changes matrix to:
        //    A = [ 2     0     0     8
        //          1     2     1     0
        //          0    -1     2     1
        //          0     4     1     2];
        auto tMatrixEntries =
            Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tMatrix.entries());
        tMatrixEntries[6] = 2.0;
        tMatrixEntries[10] = 2.0;
        Kokkos::deep_copy(tMatrix.entries(), tMatrixEntries);

        const auto tExpected = std::vector{-2.375, 1.09375, 2.1875, 0.71875};
        const auto tResult = solve_and_check_solution(tMatrix, tExpected);

        TEST_ASSERT(tResult.first);
        out << tResult.second;
    }
    // Change sparsity pattern
    {
        // Changes matrix to:
        //    A = [ 2     0     8     0
        //          1     2     1     0
        //          0    -1     2     1
        //          4     0     1     2];
        auto tColumns =
            Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tMatrix.columnIndices());
        tColumns[1] = 2;
        tColumns[8] = 0;
        Kokkos::deep_copy(tMatrix.columnIndices(), tColumns);

        const auto tExpected = std::vector{-0.375, 1.078125, 0.21875, 4.640625};
        const auto tResult = solve_and_check_solution(tMatrix, tExpected);

        TEST_ASSERT(tResult.first);
        out << tResult.second;
    }
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, ThrowsForSingularMatrix)
{
    TEST_ASSERT(!std::filesystem::exists(Plato::alg::bad_umfpack_matrix_file_path()));

    auto tMatrix = asymmetric_matrix();
    // Changes matrix to:
    //    A = [ 0     0     0     8
    //          1     2     1     0
    //          0    -1    -2     1
    //          0     4     1    -2];
    auto tMatrixEntries = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tMatrix.entries());
    tMatrixEntries[0] = 0.0;
    tMatrixEntries[1] = 0.0;
    Kokkos::deep_copy(tMatrix.entries(), tMatrixEntries);

    const auto tExpected = std::vector{0.0, 0.0, 0.0, 0.0};
    TEST_THROW([[maybe_unused]] const auto tResult = solve_and_check_solution(tMatrix, tExpected), std::runtime_error);

    TEST_ASSERT(std::filesystem::exists(Plato::alg::bad_umfpack_matrix_file_path()));
    std::filesystem::remove(Plato::alg::bad_umfpack_matrix_file_path());
}
