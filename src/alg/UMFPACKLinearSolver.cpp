#include <iostream>
#include "UMFPACKLinearSolver.hpp"

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
    return CSRMatrix{kokkosViewToStdVector<SuiteSparse_long>(aA.rowMap()),
                     kokkosViewToStdVector<SuiteSparse_long>(aA.columnIndices()),
                     kokkosViewToStdVector<double>(aA.entries())};
}

UMFPACKLinearSolver::UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs) :
                                         Plato::AbstractSolver(aSolverParams, aMPCs)
{
}

UMFPACKLinearSolver::~UMFPACKLinearSolver() {
    if (Symbolic != nullptr) {
        umfpack_dl_free_symbolic(&Symbolic);
    }
    if (Numeric != nullptr) {
        umfpack_dl_free_numeric(&Numeric);
    }
}

void UMFPACKLinearSolver::innerSolve(Plato::CrsMatrix<int> aA, Plato::ScalarVector aX, Plato::ScalarVector aB)
{
    const CSRMatrix A = constructCSRMatrix(aA);

    mMatrix = convertCSRtoCSC(A);
    const SuiteSparse_long nRows = mMatrix.nCols();

    if (Symbolic == nullptr) {
        umfpack_dl_symbolic(nRows, nRows, mMatrix.colBegin.data(), mMatrix.rows.data(), mMatrix.values.data(), &Symbolic, nullptr, Info.data());
        check_umfpack("symbolic factorization");
    }
    if (Numeric == nullptr) {
        umfpack_dl_numeric(mMatrix.colBegin.data(), mMatrix.rows.data(), mMatrix.values.data(), Symbolic, &Numeric, nullptr, Info.data());
        check_umfpack("numeric factorization");
    }

    umfpack_dl_solve(UMFPACK_A, mMatrix.colBegin.data(), mMatrix.rows.data(), mMatrix.values.data(), aX.data(), aB.data(), Numeric, nullptr, Info.data());
    check_umfpack("matrix solve");
}

void UMFPACKLinearSolver::report_memory_usage() {
    std::cout << "UMFPACK peak memory usage: " << Info[UMFPACK_SIZE_OF_UNIT]*Info[UMFPACK_PEAK_MEMORY]/(1024.0*1024.0) << " MB." << std::endl;
}

void UMFPACKLinearSolver::check_umfpack(const char *msg) {
    if (Info[UMFPACK_STATUS] != UMFPACK_OK) {
        std::cerr << "UMFPACK: error in " << msg << ": status = " << Info[UMFPACK_STATUS] << std::endl;
    }
}

} // namespace Plato::UMFPACK