#include "alg/CHOLMODLinearSolver.hpp"

#include "CrsMatrixUtils.hpp"
#include "alg/SuiteSparseUtils.hpp"

namespace Plato::alg
{
namespace
{
using PlatoOrdinalType = int;
using CholmodIndexType = std::int32_t;

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

CholmodCommonSetupTeardown::CholmodCommonSetupTeardown(const Plato::LinearSystemType aLinearSystemType)
{
    cholmod_start(&mValue);
    mValue.supernodal =
        aLinearSystemType != LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE ? CHOLMOD_SIMPLICIAL : CHOLMOD_AUTO;
}

CholmodCommonSetupTeardown::~CholmodCommonSetupTeardown() { cholmod_finish(&mValue); }

CholmodFactorSetupTeardown::CholmodFactorSetupTeardown(
    cholmod_sparse *const aCholmodSparse, std::reference_wrapper<CholmodCommonSetupTeardown> &&aCholmodCommon)
    : mCholmodCommon{std::move(aCholmodCommon)}, mValue{cholmod_analyze(aCholmodSparse, &aCholmodCommon.get().mValue)}
{
}

CholmodFactorSetupTeardown::~CholmodFactorSetupTeardown()
{
    if (mValue)
    {
        cholmod_free_factor(&mValue, &mCholmodCommon.get().mValue);
    }
}

CholmodFactorSetupTeardown::CholmodFactorSetupTeardown(CholmodFactorSetupTeardown &&aOther) noexcept
    : mValue{aOther.mValue}, mCholmodCommon{aOther.mCholmodCommon}
{
    aOther.mValue = nullptr;
}

auto CholmodFactorSetupTeardown::operator=(CholmodFactorSetupTeardown &&aOther) noexcept -> CholmodFactorSetupTeardown &
{
    if (this != &aOther)
    {
        mValue = aOther.mValue;
        mCholmodCommon = aOther.mCholmodCommon;
        aOther.mValue = nullptr;
    }
    return *this;
}

CHOLMODLinearSolver::CHOLMODLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         const Plato::LinearSystemType aLinearSystemType,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs)
    : Plato::AbstractSolver(aSolverParams, aMPCs),
      mCholmodCommon{aLinearSystemType},
      mCholmodFactorCache{[this](const CSRMatrix &, cholmod_sparse *const aCholmodSparse)
                          { return CholmodFactorSetupTeardown{aCholmodSparse, std::ref(mCholmodCommon)}; },
                          [](const CSRMatrix &aMatrix, cholmod_sparse *const)
                          { return crs_matrix_row_column_hash(aMatrix.mRowBegin, aMatrix.mColumns); }}
{
}

void CHOLMODLinearSolver::innerSolve(const Plato::CrsMatrixType aA,
                                     const Plato::ScalarVector aX,
                                     const Plato::ScalarVector aB)
{
    std::cout << "Using CHOLMOD\n";
    const auto [tRowBegin, tColumns, tValues] = crs_matrix_non_block_form<Plato::OrdinalType>(aA);
    if (!has_symmetric_sparsity_pattern<PlatoOrdinalType>(tRowBegin, tColumns))
    {
        throw std::runtime_error(
            "CHOLMOD was given a matrix with a non-symmetric sparsity pattern.\n"
            "CHOLMOD must only be used with symmetric matrices, for general matrices use UMFPACK.");
    }

    const auto tCRSMatrix = constructCSRMatrix(tRowBegin, tColumns, tValues);
    auto *tCholmodSparseA = convertSymmetricCSRtoCHOLMODSparse(tCRSMatrix, &mCholmodCommon.mValue);

    const auto &tCholmodFactor = mCholmodFactorCache.compute(tCRSMatrix, tCholmodSparseA);
    cholmod_factorize(tCholmodSparseA, tCholmodFactor.mValue, &mCholmodCommon.mValue);

    auto tRHS = CHOLMODVector{aB, &mCholmodCommon.mValue};
    auto tSolution = cholmod_solve(CHOLMOD_A, tCholmodFactor.mValue, tRHS.get(), &mCholmodCommon.mValue);

    cholmod_to_scalar_vector(*tSolution, aX);
    cholmod_free_dense(&tSolution, &mCholmodCommon.mValue);
}

auto convertSymmetricCSRtoCHOLMODSparse(const CSRMatrix &aMatrix,
                                        cholmod_common *const aCholmodCommon) -> cholmod_sparse *
{
    constexpr auto tCholmodSTypeLowerDiagonal = -1;
    auto *tCholmodTriplet =
        cholmod_allocate_triplet(aMatrix.numberOfRows(), aMatrix.numberOfRows(), aMatrix.mValues.size(),
                                 tCholmodSTypeLowerDiagonal, CHOLMOD_REAL, aCholmodCommon);

    auto tCholmodCounter = CholmodIndexType{0};
    for (auto tRowIndex = 0; tRowIndex < aMatrix.mRowBegin.size() - 1; ++tRowIndex)
    {
        const auto tStartIndex = aMatrix.mRowBegin[tRowIndex];
        const auto tEndIndex = aMatrix.mRowBegin[tRowIndex + 1];
        for (auto tIndexIntoEntries = tStartIndex; tIndexIntoEntries < tEndIndex; ++tIndexIntoEntries)
        {
            const auto tColIndex = aMatrix.mColumns[tIndexIntoEntries];
            if (tColIndex <= tRowIndex)
            {
                static_cast<CholmodIndexType *>(tCholmodTriplet->i)[tCholmodCounter] = tRowIndex;
                static_cast<CholmodIndexType *>(tCholmodTriplet->j)[tCholmodCounter] = tColIndex;
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

}  // namespace Plato::alg
