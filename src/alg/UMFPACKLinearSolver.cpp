#include "UMFPACKLinearSolver.hpp"

#include <umfpack.h>

#include <iostream>

#include "CrsMatrixUtils.hpp"

namespace Plato::alg
{
namespace
{
/// @brief Used for custom deleter of UMFPACK numeric object.
struct UMFPACKNumericDeleter
{
    void operator()(void *aUMFPACKSymbolic);
};

using UMFPACKNumeric = std::unique_ptr<void, UMFPACKNumericDeleter>;

void UMFPACKNumericDeleter::operator()(void *aUMFPACKNumeric)
{
    if (aUMFPACKNumeric)
    {
        umfpack_dl_free_numeric(&aUMFPACKNumeric);
    }
}

void check_umfpack(const std::string &aMessage,
                   const std::array<double, UMFPACK_INFO> &aInfo,
                   const Plato::CrsMatrix<Plato::OrdinalType> &aMatrix)
{
    if (aInfo[UMFPACK_STATUS] != UMFPACK_OK)
    {
        const auto [tRowEntrySpans, tColumns, tValues] = crs_matrix_non_block_form(aMatrix);
        print_matrix_to_file<Plato::OrdinalType>(tRowEntrySpans, tColumns, tValues, bad_umfpack_matrix_file_path());
        ANALYZE_THROWERR("UMFPACK: error in " + aMessage +
                         ": status = " + std::to_string(static_cast<int>(aInfo[UMFPACK_STATUS])));
    }
}

[[nodiscard]] auto numeric_factorization(const CCSMatrix &aCCSMatrix,
                                         const CrsMatrixType &aCRSMatrix,
                                         void *const aUMFPACKSymbolic) -> UMFPACKNumeric
{
    auto tInfo = std::array<double, UMFPACK_INFO>{};
    void *tNumeric = nullptr;
    umfpack_dl_numeric(aCCSMatrix.mColumnBegin.data(), aCCSMatrix.mRows.data(), aCCSMatrix.mValues.data(),
                       aUMFPACKSymbolic, &tNumeric, nullptr, tInfo.data());
    auto tWrappedNumeric = UMFPACKNumeric{tNumeric};
    check_umfpack("Numeric factorization", tInfo, aCRSMatrix);
    return tWrappedNumeric;
}

}  // namespace

void UMFPACKSymbolicDeleter::operator()(void *aUMFPACKSymbolic)
{
    if (aUMFPACKSymbolic)
    {
        umfpack_dl_free_symbolic(&aUMFPACKSymbolic);
    }
}

UMFPACKLinearSolver::UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs)
    : Plato::AbstractSolver(aSolverParams, aMPCs),
      mUMFPACKSymbolicCache{[](const CCSMatrix &aMatrix, const CrsMatrixType &aCRSMatrix)
                            {
                                void *tSymbolic = nullptr;
                                const auto tNumberOfRows = aMatrix.numberOfColumns();
                                auto tInfo = std::array<double, UMFPACK_INFO>{};
                                umfpack_dl_symbolic(tNumberOfRows, tNumberOfRows, aMatrix.mColumnBegin.data(),
                                                    aMatrix.mRows.data(), aMatrix.mValues.data(), &tSymbolic, nullptr,
                                                    tInfo.data());
                                auto tWrappedSymbolic = UMFPACKSymbolic{tSymbolic};
                                check_umfpack("Symbolic factorization", tInfo, aCRSMatrix);
                                return tWrappedSymbolic;
                            },
                            [](const CCSMatrix &aMatrix, const CrsMatrixType &)
                            { return crs_matrix_row_column_hash(aMatrix.mColumnBegin, aMatrix.mRows); }}
{
}

void UMFPACKLinearSolver::innerSolve(Plato::CrsMatrix<Plato::OrdinalType> aCRSMatrix,
                                     Plato::ScalarVector aX,
                                     Plato::ScalarVector aB)
{
    const auto tNumberOfRows = aCRSMatrix.numRows();
    auto tMatrix = to_CCS(make_CRS_matrix(aCRSMatrix));

    const auto &tSymbolic = mUMFPACKSymbolicCache.compute(tMatrix, aCRSMatrix);
    const auto tNumeric = numeric_factorization(tMatrix, aCRSMatrix, tSymbolic.get());

    const auto tRHSOnHost = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, aB);
    const auto tSolutionOnHost = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, aX);
    auto tInfo = std::array<double, UMFPACK_INFO>{};
    umfpack_dl_solve(UMFPACK_A, tMatrix.mColumnBegin.data(), tMatrix.mRows.data(), tMatrix.mValues.data(),
                     tSolutionOnHost.data(), tRHSOnHost.data(), tNumeric.get(), nullptr, tInfo.data());
    check_umfpack("matrix solve", tInfo, aCRSMatrix);

    Kokkos::deep_copy(aX, tSolutionOnHost);
}

auto bad_umfpack_matrix_file_path() -> std::filesystem::path { return std::filesystem::path{"bad_umfpack_matrix.m"}; }

}  // namespace Plato::alg
