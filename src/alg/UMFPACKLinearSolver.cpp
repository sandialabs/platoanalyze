#include <iostream>
#include "UMFPACKLinearSolver.hpp"
#include "CrsMatrixUtils.hpp"

namespace Plato::UMFPACK {

CSCMatrix convertCSRtoCSC(const CSRMatrix &A)
{
    assert(A.rowBegin.size() > 0);
    assert(A.columns.size() == A.values.size());
    const SuiteSparse_long nRows = A.nRows();
    const SuiteSparse_long nEntries = A.columns.size();

    std::vector<SuiteSparse_long> rows(nEntries);

    if (UMFPACK_OK != umfpack_dl_col_to_triplet(nRows, A.rowBegin.data(), rows.data())) {
        ANALYZE_THROWERR("Column to triplet conversion failed.");
    }

    CSCMatrix B;
    B.colBegin.resize(nRows+1);
    B.rows.resize(nEntries);
    B.values.resize(nEntries);

    if (UMFPACK_OK != umfpack_dl_triplet_to_col(nRows, nRows, nEntries,
                                                rows.data(), A.columns.data(), A.values.data(),
                                                B.colBegin.data(), B.rows.data(), B.values.data(),
                                                nullptr)) {
        ANALYZE_THROWERR("Triplet to column conversion failed.");
    }

    return B;
}

namespace {
template <typename ReturnType,typename ViewType>
std::vector<ReturnType> kokkosViewToStdVector(ViewType v) {
    std::vector<ReturnType> vec;

    static_assert(ViewType::rank() == 1, "invalid usage of kokkosViewToStdVector: requires one dimension");

    vec.reserve(v.size());
    std::copy(v.data(), v.data() + v.size(), std::back_inserter(vec));

    return vec;
}
}

CSRMatrix constructCSRMatrix(const Plato::CrsMatrix<int> &aA)
{
    using CrsOrdinal = int;
    Plato::CrsMatrix<CrsOrdinal>::RowMapVectorT tRowBegin;
    Plato::CrsMatrix<CrsOrdinal>::OrdinalVectorT tColumns;
    Plato::CrsMatrix<CrsOrdinal>::ScalarVectorT tValues;
    std::tie(tRowBegin, tColumns, tValues) = Plato::crs_matrix_non_block_form<CrsOrdinal>(aA);

    return CSRMatrix{kokkosViewToStdVector<SuiteSparse_long>(tRowBegin),
                     kokkosViewToStdVector<SuiteSparse_long>(tColumns),
                     kokkosViewToStdVector<double>(tValues)};
}

UMFPACKLinearSolver::UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs) :
                                         Plato::AbstractSolver(aSolverParams, aMPCs)
{
}

void UMFPACKLinearSolver::clear() {
    if (mSymbolic != nullptr) {
        umfpack_dl_free_symbolic(&mSymbolic);
        mSymbolic = nullptr;
    }
    if (mNumeric != nullptr) {
        umfpack_dl_free_numeric(&mNumeric);
        mNumeric = nullptr;
    }

}

void UMFPACKLinearSolver::innerSolve(Plato::CrsMatrix<int> aA,
                                     Plato::ScalarVector aX,
                                     Plato::ScalarVector aB)
{
    const CSRMatrix A = constructCSRMatrix(aA);

    mMatrix = convertCSRtoCSC(A);
    const SuiteSparse_long nRows = mMatrix.nCols();

    umfpack_dl_symbolic(nRows, nRows, mMatrix.colBegin.data(), mMatrix.rows.data(), mMatrix.values.data(), &mSymbolic, nullptr, mInfo.data());
    check_umfpack("Symbolic factorization");

    umfpack_dl_numeric(mMatrix.colBegin.data(), mMatrix.rows.data(), mMatrix.values.data(), mSymbolic, &mNumeric, nullptr, mInfo.data());
    check_umfpack("Numeric factorization");

    umfpack_dl_solve(UMFPACK_A, mMatrix.colBegin.data(), mMatrix.rows.data(), mMatrix.values.data(), aX.data(), aB.data(), mNumeric, nullptr, mInfo.data());
    check_umfpack("matrix solve");

    report_memory_usage();

    clear();
}

void UMFPACKLinearSolver::report_memory_usage() {
    std::cout << "UMFPACK peak memory usage: " << mInfo[UMFPACK_SIZE_OF_UNIT]*mInfo[UMFPACK_PEAK_MEMORY]/(1024.0*1024.0) << " MB." << std::endl;
}

void UMFPACKLinearSolver::check_umfpack(const std::string &msg) {
    if (mInfo[UMFPACK_STATUS] != UMFPACK_OK) {
        ANALYZE_THROWERR("UMFPACK: error in " + msg + ": status = " + std::to_string(static_cast<int>(mInfo[UMFPACK_STATUS])));
    }
}

} // namespace Plato::UMFPACK