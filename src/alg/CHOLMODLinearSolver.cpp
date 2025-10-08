#include "alg/CHOLMODLinearSolver.hpp"

#include "CrsMatrixUtils.hpp"
#include "alg/SuiteSparseUtils.hpp"

namespace Plato::alg
{
namespace
{
using PlatoOrdinalType = int;
using kCholmodIndexType = std::int32_t;

[[nodiscard]] auto convertCSRtoCHOLMODSparse(const CSRMatrix &aMatrix, cholmod_common *const aCholmodCommon)
{
    constexpr auto tCholmodSTypeLowerDiagonal = -1;
    auto *tCholmodTriplet =
        cholmod_allocate_triplet(aMatrix.numberOfRows(), aMatrix.numberOfRows(), aMatrix.mValues.size(),
                                 tCholmodSTypeLowerDiagonal, CHOLMOD_REAL, aCholmodCommon);

    auto tCholmodCounter = kCholmodIndexType{0};
    for (auto tRowIndex = 0; tRowIndex < aMatrix.mRowBegin.size() - 1; ++tRowIndex)
    {
        const auto tStartIndex = aMatrix.mRowBegin[tRowIndex];
        const auto tEndIndex = aMatrix.mRowBegin[tRowIndex + 1];
        for (auto tIndexIntoEntries = tStartIndex; tIndexIntoEntries < tEndIndex; ++tIndexIntoEntries)
        {
            const auto tColIndex = aMatrix.mColumns[tIndexIntoEntries];
            if (tColIndex <= tRowIndex)
            {
                static_cast<kCholmodIndexType *>(tCholmodTriplet->i)[tCholmodCounter] = tRowIndex;
                static_cast<kCholmodIndexType *>(tCholmodTriplet->j)[tCholmodCounter] = tColIndex;
                static_cast<double *>(tCholmodTriplet->x)[tCholmodCounter] = aMatrix.mValues[tIndexIntoEntries];
                ++tCholmodCounter;
            }
        }
    }

    tCholmodTriplet->nnz = tCholmodCounter;
    auto *tCholmodSparse = cholmod_triplet_to_sparse(tCholmodTriplet, tCholmodCounter, aCholmodCommon);
    cholmod_free_triplet(&tCholmodTriplet, aCholmodCommon);
    return tCholmodSparse;
}

/// @brief A wrapper for a `cholmod_dense` struct representing a vector. Cleans up its allocation on destruction.
class CHOLMODVector
{
   public:
    CHOLMODVector(const Plato::ScalarVector aPlatoVector, cholmod_common *const aCholmodCommon)
        : mCholmodCommon{aCholmodCommon},
          mVector{cholmod_allocate_dense(
              aPlatoVector.size(), 1, aPlatoVector.size(), CHOLMOD_DOUBLE + CHOLMOD_REAL, aCholmodCommon)}
    {
        const auto tMirror = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, aPlatoVector);
        std::copy_n(tMirror.data(), tMirror.size(), static_cast<double *>(mVector->x));
    }

    ~CHOLMODVector() { cholmod_free_dense(&mVector, mCholmodCommon); }

    [[nodiscard]] auto get() -> cholmod_dense * { return mVector; }

   private:
    cholmod_common *mCholmodCommon;
    cholmod_dense *mVector;
};

void cholmod_to_scalar_vector(const cholmod_dense &aCHOLMODDense, const Plato::ScalarVector aPlatoVector)
{
    const auto tSolutionMirror = Kokkos::create_mirror_view(aPlatoVector);
    std::copy_n(static_cast<double *>(aCHOLMODDense.x), aCHOLMODDense.nrow, tSolutionMirror.data());
    Kokkos::deep_copy(aPlatoVector, tSolutionMirror);
}

}  // namespace

CHOLMODLinearSolver::CHOLMODLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         const Plato::LinearSystemType aLinearSystemType,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs)
    : Plato::AbstractSolver(aSolverParams, aMPCs)
{
    cholmod_start(&mCholmodCommon);
    mCholmodCommon.supernodal =
        aLinearSystemType != LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE ? CHOLMOD_SIMPLICIAL : CHOLMOD_AUTO;
}

CHOLMODLinearSolver::~CHOLMODLinearSolver()
{
    if (mCholmodFactor)
    {
        cholmod_free_factor(&mCholmodFactor, &mCholmodCommon);
    }
    cholmod_finish(&mCholmodCommon);
}

void CHOLMODLinearSolver::innerSolve(const Plato::CrsMatrix<PlatoOrdinalType> aA,
                                     const Plato::ScalarVector aX,
                                     const Plato::ScalarVector aB)
{
    const auto [tRowBegin, tColumns, tValues] = crs_matrix_non_block_form<PlatoOrdinalType>(aA);
    if (!has_symmetric_sparsity_pattern<PlatoOrdinalType>(tRowBegin, tColumns))
    {
        throw std::runtime_error(
            "CHOLMOD was given a matrix with a non-symmetric sparsity pattern.\n"
            "CHOLMOD must only be used with symmetric matrices, for general matrices use UMFPACK.");
    }
    auto *tCholmodSparseA =
        convertCSRtoCHOLMODSparse(constructCSRMatrix(tRowBegin, tColumns, tValues), &mCholmodCommon);

    const auto tNumberOfRows = aA.numRows();

    if (const auto tNewMatrixPatternHash = crs_matrix_row_column_hash<int>(aA.rowMap(), aA.columnIndices());
        !mCurrentMatrixPatternHash.has_value() || mCurrentMatrixPatternHash.value() != tNewMatrixPatternHash)
    {
        mCurrentMatrixPatternHash = tNewMatrixPatternHash;
        mCholmodFactor = cholmod_analyze(tCholmodSparseA, &mCholmodCommon);
    }
    cholmod_factorize(tCholmodSparseA, mCholmodFactor, &mCholmodCommon);

    auto tRHS = CHOLMODVector{aB, &mCholmodCommon};
    auto tSolution = cholmod_solve(CHOLMOD_A, mCholmodFactor, tRHS.get(), &mCholmodCommon);

    cholmod_to_scalar_vector(*tSolution, aX);
    cholmod_free_dense(&tSolution, &mCholmodCommon);
}

}  // namespace Plato::alg
