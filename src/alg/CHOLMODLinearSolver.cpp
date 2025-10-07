#include "alg/CHOLMODLinearSolver.hpp"

#include "alg/SuiteSparseUtils.hpp"

namespace Plato::alg
{
namespace
{
using kCholmodIndexType = std::int32_t;

[[nodiscard]] auto convertCSRtoCHOLMODSparse(const Plato::CrsMatrix<int> &aMatrix, cholmod_common *aCholmodCommon)
{
    const auto tCSRMatrix = constructCSRMatrix(aMatrix);

    auto *tCholmodTriplet = cholmod_allocate_triplet(tCSRMatrix.numberOfRows(), tCSRMatrix.numberOfRows(),
                                                     tCSRMatrix.mValues.size(), 0, CHOLMOD_REAL, aCholmodCommon);

    auto tCholmodCounter = kCholmodIndexType{0};
    for (auto tRowIndex = 0; tRowIndex < tCSRMatrix.mRowBegin.size() - 1; ++tRowIndex)
    {
        const auto tStartIndex = tCSRMatrix.mRowBegin[tRowIndex];
        const auto tEndIndex = tCSRMatrix.mRowBegin[tRowIndex + 1];
        for (auto tIndexIntoEntries = tStartIndex; tIndexIntoEntries < tEndIndex; ++tIndexIntoEntries)
        {
            const auto tColIndex = tCSRMatrix.mColumns[tIndexIntoEntries];
            if (tColIndex <= tRowIndex)
            {
                static_cast<kCholmodIndexType *>(tCholmodTriplet->i)[tCholmodCounter] = tRowIndex;
                static_cast<kCholmodIndexType *>(tCholmodTriplet->j)[tCholmodCounter] = tColIndex;
                static_cast<double *>(tCholmodTriplet->x)[tCholmodCounter] = tCSRMatrix.mValues[tIndexIntoEntries];
                ++tCholmodCounter;
            }
        }
    }

    tCholmodTriplet->stype = -1;
    tCholmodTriplet->nnz = tCholmodCounter;
    auto *tCholmodSparse = cholmod_triplet_to_sparse(tCholmodTriplet, tCholmodCounter, aCholmodCommon);
    cholmod_free_triplet(&tCholmodTriplet, aCholmodCommon);
    return tCholmodSparse;
}
}  // namespace

CHOLMODLinearSolver::CHOLMODLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs)
    : Plato::AbstractSolver(aSolverParams, aMPCs)
{
    cholmod_start(&mCholmodCommon);
}

CHOLMODLinearSolver::~CHOLMODLinearSolver()
{
    if (mCholmodFactor)
    {
        cholmod_free_factor(&mCholmodFactor, &mCholmodCommon);
    }
    cholmod_finish(&mCholmodCommon);
}

void CHOLMODLinearSolver::innerSolve(const Plato::CrsMatrix<int> aA, Plato::ScalarVector aX, Plato::ScalarVector aB)
{
    auto *tCholmodSparseA = convertCSRtoCHOLMODSparse(aA, &mCholmodCommon);

    const auto tNumberOfRows = aA.numRows();

    mCholmodFactor = cholmod_analyze(tCholmodSparseA, &mCholmodCommon);
    cholmod_factorize(tCholmodSparseA, mCholmodFactor, &mCholmodCommon);

    auto *tRHS =
        cholmod_allocate_dense(tNumberOfRows, 1, tNumberOfRows, CHOLMOD_DOUBLE + CHOLMOD_REAL, &mCholmodCommon);
    // TODO: Use a kokkos mirror for copying solution
    std::copy_n(aB.data(), aB.size(), static_cast<double *>(tRHS->x));

    auto tSolution = cholmod_solve(CHOLMOD_A, mCholmodFactor, tRHS, &mCholmodCommon);

    // TODO: Use a kokkos mirror for copying solution
    std::copy_n(static_cast<double *>(tSolution->x), tNumberOfRows, aX.data());

    cholmod_free_dense(&tSolution, &mCholmodCommon);
    cholmod_free_dense(&tRHS, &mCholmodCommon);
}

}  // namespace Plato::alg
