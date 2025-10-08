#include "UMFPACKLinearSolver.hpp"

#include <iostream>

#include "CrsMatrixUtils.hpp"

namespace Plato::alg
{
UMFPACKLinearSolver::UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs)
    : Plato::AbstractSolver(aSolverParams, aMPCs)
{
}

void UMFPACKLinearSolver::clear()
{
    if (mSymbolic != nullptr)
    {
        umfpack_dl_free_symbolic(&mSymbolic);
        mSymbolic = nullptr;
    }
    if (mNumeric != nullptr)
    {
        umfpack_dl_free_numeric(&mNumeric);
        mNumeric = nullptr;
    }
}

void UMFPACKLinearSolver::innerSolve(Plato::CrsMatrix<int> aA, Plato::ScalarVector aX, Plato::ScalarVector aB)
{
    /*
        umfpack_dl_symbolic(tNumberOfRows, tNumberOfRows, mMatrix.colBegin.data(), mMatrix.rows.data(),
       mMatrix.values.data(), &mSymbolic, nullptr, mInfo.data()); check_umfpack("Symbolic factorization");

        umfpack_dl_numeric(mMatrix.colBegin.data(), mMatrix.rows.data(), mMatrix.values.data(), mSymbolic, &mNumeric,
                           nullptr, mInfo.data());
        check_umfpack("Numeric factorization");

        umfpack_dl_solve(UMFPACK_A, mMatrix.colBegin.data(), mMatrix.rows.data(), mMatrix.values.data(), aX.data(),
                         aB.data(), mNumeric, nullptr, mInfo.data());
        check_umfpack("matrix solve");

        report_memory_usage();

        clear();
    */
}

void UMFPACKLinearSolver::report_memory_usage()
{
    std::cout << "UMFPACK peak memory usage: "
              << mInfo[UMFPACK_SIZE_OF_UNIT] * mInfo[UMFPACK_PEAK_MEMORY] / (1024.0 * 1024.0) << " MB." << std::endl;
}

void UMFPACKLinearSolver::check_umfpack(const std::string &msg)
{
    if (mInfo[UMFPACK_STATUS] != UMFPACK_OK)
    {
        ANALYZE_THROWERR("UMFPACK: error in " + msg +
                         ": status = " + std::to_string(static_cast<int>(mInfo[UMFPACK_STATUS])));
    }
}

}  // namespace Plato::alg
