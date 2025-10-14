#include "alg/CHOLMODLinearSolver.hpp"

#include "CrsMatrixUtils.hpp"
#include "alg/SuiteSparseUtils.hpp"

namespace Plato::alg
{
namespace
{
using CHOLMODIndexType = std::int32_t;

template <typename CHOLMODObject>
[[nodiscard]] auto make_cholmod_wrapper(CHOLMODObject *const aCHOLMODObject,
                                        std::reference_wrapper<CHOLMODCommonSetupTeardown> &&aCHOLMODCommon)
    -> CHOLMODObjectWrapper<CHOLMODObject>
{
    if constexpr (std::is_same_v<CHOLMODObject, cholmod_sparse>)
    {
        return CHOLMODObjectWrapper{aCHOLMODObject, std::move(aCHOLMODCommon), cholmod_free_sparse};
    }
    else if constexpr (std::is_same_v<CHOLMODObject, cholmod_dense>)
    {
        return CHOLMODObjectWrapper{aCHOLMODObject, std::move(aCHOLMODCommon), cholmod_free_dense};
    }
    else if constexpr (std::is_same_v<CHOLMODObject, cholmod_triplet>)
    {
        return CHOLMODObjectWrapper{aCHOLMODObject, std::move(aCHOLMODCommon), cholmod_free_triplet};
    }
}

[[nodiscard]] auto make_cholmod_factor_wrapper(cholmod_sparse *const aCHOLMODSparse,
                                               std::reference_wrapper<CHOLMODCommonSetupTeardown> &&aCHOLMODCommon)
    -> CHOLMODObjectWrapper<cholmod_factor>
{
    return CHOLMODObjectWrapper<cholmod_factor>{cholmod_analyze(aCHOLMODSparse, &aCHOLMODCommon.get().mValue),
                                                std::move(aCHOLMODCommon), cholmod_free_factor};
}

/// @brief A wrapper for a `cholmod_dense` struct representing a vector. Cleans up its allocation on destruction.
class CHOLMODVector
{
   public:
    CHOLMODVector(const Plato::ScalarVector aPlatoVector, cholmod_common *const aCHOLMODCommon)
        : mCHOLMODCommon{aCHOLMODCommon},
          mVector{cholmod_allocate_dense(
              aPlatoVector.size(), 1, aPlatoVector.size(), CHOLMOD_DOUBLE + CHOLMOD_REAL, aCHOLMODCommon)}
    {
        const auto tMirror = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, aPlatoVector);
        std::copy_n(tMirror.data(), tMirror.size(), static_cast<double *>(mVector->x));
    }

    ~CHOLMODVector() { cholmod_free_dense(&mVector, mCHOLMODCommon); }

    [[nodiscard]] auto get() -> cholmod_dense * { return mVector; }

   private:
    cholmod_common *mCHOLMODCommon;
    cholmod_dense *mVector;
};

void cholmod_to_scalar_vector(const cholmod_dense &aCHOLMODDense, const Plato::ScalarVector aPlatoVector)
{
    const auto tSolutionMirror = Kokkos::create_mirror_view(aPlatoVector);
    std::copy_n(static_cast<double *>(aCHOLMODDense.x), aCHOLMODDense.nrow, tSolutionMirror.data());
    Kokkos::deep_copy(aPlatoVector, tSolutionMirror);
}

}  // namespace

CHOLMODCommonSetupTeardown::CHOLMODCommonSetupTeardown(const Plato::LinearSystemType aLinearSystemType)
{
    cholmod_start(&mValue);
    mValue.supernodal =
        aLinearSystemType != LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE ? CHOLMOD_SIMPLICIAL : CHOLMOD_AUTO;
}

CHOLMODCommonSetupTeardown::~CHOLMODCommonSetupTeardown() { cholmod_finish(&mValue); }

CHOLMODLinearSolver::CHOLMODLinearSolver(const Teuchos::ParameterList &aSolverParams,
                                         const Plato::LinearSystemType aLinearSystemType,
                                         std::shared_ptr<Plato::MultipointConstraints> aMPCs)
    : Plato::AbstractSolver(aSolverParams, aMPCs),
      mCHOLMODCommon{aLinearSystemType},
      mCHOLMODFactorCache{[this](const CSRMatrix &, cholmod_sparse *const aCHOLMODSparse)
                          { return make_cholmod_factor_wrapper(aCHOLMODSparse, std::ref(mCHOLMODCommon)); },
                          [](const CSRMatrix &aMatrix, cholmod_sparse *const)
                          { return crs_matrix_row_column_hash(aMatrix.mRowBegin, aMatrix.mColumns); }}
{
}

void CHOLMODLinearSolver::innerSolve(const Plato::CrsMatrixType aA,
                                     const Plato::ScalarVector aX,
                                     const Plato::ScalarVector aB)
{
    const auto [tRowBegin, tColumns, tValues] = crs_matrix_non_block_form<Plato::OrdinalType>(aA);
    if (!has_symmetric_sparsity_pattern<Plato::OrdinalType>(tRowBegin, tColumns))
    {
        throw std::runtime_error(
            "CHOLMOD was given a matrix with a non-symmetric sparsity pattern.\n"
            "CHOLMOD must only be used with symmetric matrices, for general matrices use UMFPACK.");
    }

    const auto tCRSMatrix = make_CSR_matrix(tRowBegin, tColumns, tValues);
    auto tCHOLMODSparseA = convert_symmetric_CSR_to_CHOLMOD_sparse(tCRSMatrix, mCHOLMODCommon);

    const auto &tCHOLMODFactor = mCHOLMODFactorCache.compute(tCRSMatrix, tCHOLMODSparseA.mObject);
    cholmod_factorize(tCHOLMODSparseA.mObject, tCHOLMODFactor.mObject, &mCHOLMODCommon.mValue);

    auto tRHS = CHOLMODVector{aB, &mCHOLMODCommon.mValue};
    auto tSolution = make_cholmod_wrapper(
        cholmod_solve(CHOLMOD_A, tCHOLMODFactor.mObject, tRHS.get(), &mCHOLMODCommon.mValue), std::ref(mCHOLMODCommon));

    cholmod_to_scalar_vector(*tSolution.mObject, aX);
}

auto convert_symmetric_CSR_to_CHOLMOD_sparse(const CSRMatrix &aMatrix, CHOLMODCommonSetupTeardown &aCHOLMODCommon)
    -> CHOLMODObjectWrapper<cholmod_sparse>
{
    constexpr auto tCHOLMODSTypeLowerDiagonal = -1;
    auto tCHOLMODTriplet = make_cholmod_wrapper(
        cholmod_allocate_triplet(aMatrix.numberOfRows(), aMatrix.numberOfRows(), aMatrix.mValues.size(),
                                 tCHOLMODSTypeLowerDiagonal, CHOLMOD_REAL, &aCHOLMODCommon.mValue),
        std::ref(aCHOLMODCommon));

    auto tCHOLMODCounter = CHOLMODIndexType{0};
    for (auto tRowIndex = 0; tRowIndex < aMatrix.mRowBegin.size() - 1; ++tRowIndex)
    {
        const auto tStartIndex = aMatrix.mRowBegin[tRowIndex];
        const auto tEndIndex = aMatrix.mRowBegin[tRowIndex + 1];
        for (auto tIndexIntoEntries = tStartIndex; tIndexIntoEntries < tEndIndex; ++tIndexIntoEntries)
        {
            const auto tColIndex = aMatrix.mColumns[tIndexIntoEntries];
            if (tColIndex <= tRowIndex)
            {
                static_cast<CHOLMODIndexType *>(tCHOLMODTriplet.mObject->i)[tCHOLMODCounter] = tRowIndex;
                static_cast<CHOLMODIndexType *>(tCHOLMODTriplet.mObject->j)[tCHOLMODCounter] = tColIndex;
                static_cast<double *>(tCHOLMODTriplet.mObject->x)[tCHOLMODCounter] = aMatrix.mValues[tIndexIntoEntries];
                ++tCHOLMODCounter;
            }
        }
    }

    tCHOLMODTriplet.mObject->nnz = tCHOLMODCounter;
    return make_cholmod_wrapper(
        cholmod_triplet_to_sparse(tCHOLMODTriplet.mObject, tCHOLMODCounter, &aCHOLMODCommon.mValue),
        std::ref(aCHOLMODCommon));
}

}  // namespace Plato::alg
