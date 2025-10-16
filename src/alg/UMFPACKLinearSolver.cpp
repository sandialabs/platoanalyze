#include "UMFPACKLinearSolver.hpp"

#include <umfpack.h>

#include <iostream>

#include "CrsMatrixUtils.hpp"

namespace Plato::alg
{
namespace
{
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
}  // namespace

void UMFPACKSymbolicDeleter::operator()(void *aUMFPACKSymbolic) { umfpack_dl_free_symbolic(&aUMFPACKSymbolic); }

UMFPACKLinearSolver::UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs)
    : Plato::AbstractSolver(aSolverParams, aMPCs),
      mUMFPACKSymbolicCache{[](const CSCMatrix &aMatrix, const CrsMatrixType &aCRSMatrix)
                            {
                                void *tSymbolic = nullptr;
                                const auto tNumberOfRows = aMatrix.numberOfColumns();
                                auto tInfo = std::array<double, UMFPACK_INFO>{};
                                umfpack_dl_symbolic(tNumberOfRows, tNumberOfRows, aMatrix.mColumnBegin.data(),
                                                    aMatrix.mRows.data(), aMatrix.mValues.data(), &tSymbolic, nullptr,
                                                    tInfo.data());

                                check_umfpack("Symbolic factorization", tInfo, aCRSMatrix);
                                return UMFPACKSymbolic{tSymbolic};
                            },
                            [](const CSCMatrix &aMatrix, const CrsMatrixType &)
                            { return crs_matrix_row_column_hash(aMatrix.mColumnBegin, aMatrix.mRows); }}
{
}

void UMFPACKLinearSolver::innerSolve(Plato::CrsMatrix<Plato::OrdinalType> aA,
                                     Plato::ScalarVector aX,
                                     Plato::ScalarVector aB)
{
    const auto tNumberOfRows = aA.numRows();
    auto tMatrix = to_CSC(make_CSR_matrix(aA));

    auto tInfo = std::array<double, UMFPACK_INFO>{};

    const auto &tSymbolic = mUMFPACKSymbolicCache.compute(tMatrix, aA);
    void *tNumeric = nullptr;
    umfpack_dl_numeric(tMatrix.mColumnBegin.data(), tMatrix.mRows.data(), tMatrix.mValues.data(), tSymbolic.get(),
                       &tNumeric, nullptr, tInfo.data());
    check_umfpack("Numeric factorization", tInfo, aA);

    umfpack_dl_solve(UMFPACK_A, tMatrix.mColumnBegin.data(), tMatrix.mRows.data(), tMatrix.mValues.data(), aX.data(),
                     aB.data(), tNumeric, nullptr, tInfo.data());
    check_umfpack("matrix solve", tInfo, aA);

    umfpack_dl_free_numeric(&tNumeric);
}

auto bad_umfpack_matrix_file_path() -> std::filesystem::path { return std::filesystem::path{"bad_umfpack_matrix.m"}; }

}  // namespace Plato::alg
