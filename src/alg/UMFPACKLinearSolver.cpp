#include "UMFPACKLinearSolver.hpp"

#include <umfpack.h>

#include <iostream>

#include "CrsMatrixUtils.hpp"

namespace Plato::alg
{
namespace
{
void check_umfpack(const std::string &aMessage, const std::array<double, UMFPACK_INFO> &aInfo)
{
    if (aInfo[UMFPACK_STATUS] != UMFPACK_OK)
    {
        ANALYZE_THROWERR("UMFPACK: error in " + aMessage +
                         ": status = " + std::to_string(static_cast<int>(aInfo[UMFPACK_STATUS])));
    }
}
}  // namespace

void UMFPACKSymbolicDeleter::operator()(void *aUMFPACKSymbolic) { umfpack_dl_free_symbolic(&aUMFPACKSymbolic); }

UMFPACKLinearSolver::UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs)
    : Plato::AbstractSolver(aSolverParams, aMPCs),
      mUMFPACKSymbolicCache{[](const CSCMatrix &aMatrix)
                            {
                                void *tSymbolic = nullptr;
                                const auto tNumberOfRows = aMatrix.numberOfColumns();
                                auto tInfo = std::array<double, UMFPACK_INFO>{};
                                umfpack_dl_symbolic(tNumberOfRows, tNumberOfRows, aMatrix.mColumnBegin.data(),
                                                    aMatrix.mRows.data(), aMatrix.mValues.data(), &tSymbolic, nullptr,
                                                    tInfo.data());

                                check_umfpack("Symbolic factorization", tInfo);
                                return UMFPACKSymbolic{tSymbolic};
                            },
                            [](const CSCMatrix &) { return std::size_t{0}; }}
{
}

void UMFPACKLinearSolver::innerSolve(Plato::CrsMatrix<int> aA, Plato::ScalarVector aX, Plato::ScalarVector aB)
{
    std::cout << "Using UMFPACK\n";
    const auto tNumberOfRows = aA.numRows();
    auto tMatrix = to_CSC(make_CSR_matrix(aA));

    auto tInfo = std::array<double, UMFPACK_INFO>{};

    const auto &tSymbolic = mUMFPACKSymbolicCache.compute(tMatrix);
    void *tNumeric = nullptr;
    umfpack_dl_numeric(tMatrix.mColumnBegin.data(), tMatrix.mRows.data(), tMatrix.mValues.data(), tSymbolic.get(),
                       &tNumeric, nullptr, tInfo.data());
    check_umfpack("Numeric factorization", tInfo);

    umfpack_dl_solve(UMFPACK_A, tMatrix.mColumnBegin.data(), tMatrix.mRows.data(), tMatrix.mValues.data(), aX.data(),
                     aB.data(), tNumeric, nullptr, tInfo.data());
    check_umfpack("matrix solve", tInfo);

    umfpack_dl_free_numeric(&tNumeric);
}

}  // namespace Plato::alg
