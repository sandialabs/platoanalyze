#include "solver/umfpack/CHOLMODLinearSolver.hpp"

#include "linear_algebra/CrsMatrixUtils.hpp"
#include "solver/umfpack/SuiteSparseUtils.hpp"

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

    CHOLMODVector(const CHOLMODVector &) = delete;
    CHOLMODVector(CHOLMODVector &&) = delete;
    CHOLMODVector &operator=(const CHOLMODVector &) = delete;
    CHOLMODVector &operator=(CHOLMODVector &&) = delete;

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

void check_cholmod_errors(const CHOLMODCommonSetupTeardown &aCHOLMODCommon,
                          const CrsRowsColumnsValues<Plato::OrdinalType> &aMatrix)
{
    if (aCHOLMODCommon.mValue.status != CHOLMOD_OK)
    {
        const auto &[tRowBegin, tColumns, tValues] = aMatrix;
        Plato::print_matrix_to_file<Plato::OrdinalType>(tRowBegin, tColumns, tValues,
                                                        bad_cholmod_matrix_file_path().string());
    }
    if (aCHOLMODCommon.mValue.status == CHOLMOD_NOT_POSDEF)
    {
        throw std::runtime_error{
            "CHOLMOD encountered an indefinite matrix, but expected a positive definite matrix. Aborting"};
    }
    else if (aCHOLMODCommon.mValue.status != CHOLMOD_OK)
    {
        throw std::runtime_error{"CHOLMOD encountered an error and will abort. Error code: " +
                                 std::to_string(aCHOLMODCommon.mValue.status)};
    }
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
      mCHOLMODFactorCache{[this](const CrsRowsColumnsValues<Plato::OrdinalType> &, cholmod_sparse *const aCHOLMODSparse)
                          { return make_cholmod_factor_wrapper(aCHOLMODSparse, std::ref(mCHOLMODCommon)); },
                          [](const CrsRowsColumnsValues<Plato::OrdinalType> &aMatrix, cholmod_sparse *const)
                          {
                              const auto &[tRowEntrySpans, tColumns, tValues] = aMatrix;
                              return crs_matrix_row_column_hash(tRowEntrySpans, tColumns);
                          }}
{
}

void CHOLMODLinearSolver::innerSolve(const Plato::CrsMatrixType aA,
                                     const Plato::ScalarVector aX,
                                     const Plato::ScalarVector aB)
{
    const auto tRowsColumnsAndValues = crs_matrix_non_block_form<Plato::OrdinalType>(aA);
    const auto &[tRowBegin, tColumns, tValues] = tRowsColumnsAndValues;
    if (!has_symmetric_sparsity_pattern<Plato::OrdinalType>(tRowBegin, tColumns))
    {
        throw std::runtime_error(
            "CHOLMOD was given a matrix with a non-symmetric sparsity pattern.\n"
            "CHOLMOD must only be used with symmetric matrices, for general matrices use UMFPACK.");
    }

    auto tCHOLMODSparseA = symmetric_CRS_to_CHOLMOD_sparse(tRowsColumnsAndValues, mCHOLMODCommon);

    const auto &tCHOLMODFactor = mCHOLMODFactorCache.compute(tRowsColumnsAndValues, tCHOLMODSparseA.mObject);
    check_cholmod_errors(mCHOLMODCommon, tRowsColumnsAndValues);

    cholmod_factorize(tCHOLMODSparseA.mObject, tCHOLMODFactor.mObject, &mCHOLMODCommon.mValue);
    check_cholmod_errors(mCHOLMODCommon, tRowsColumnsAndValues);

    auto tRHS = CHOLMODVector{aB, &mCHOLMODCommon.mValue};
    auto tSolution = make_cholmod_wrapper(
        cholmod_solve(CHOLMOD_A, tCHOLMODFactor.mObject, tRHS.get(), &mCHOLMODCommon.mValue), std::ref(mCHOLMODCommon));
    check_cholmod_errors(mCHOLMODCommon, tRowsColumnsAndValues);

    cholmod_to_scalar_vector(*tSolution.mObject, aX);
}

auto symmetric_CRS_to_CHOLMOD_sparse(const CrsRowsColumnsValues<Plato::OrdinalType> &aMatrix,
                                     CHOLMODCommonSetupTeardown &aCHOLMODCommon) -> CHOLMODObjectWrapper<cholmod_sparse>
{
    const auto &[tRowEntrySpansDevice, tColumnsDevice, tValuesDevice] = aMatrix;
    auto tRowEntrySpans =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tRowEntrySpansDevice);
    auto tColumns = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tColumnsDevice);
    auto tValues = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tValuesDevice);
    const auto tNumberOfRows = tRowEntrySpans.size() - 1;

    constexpr auto tCHOLMODSTypeLowerDiagonal = -1;
    auto tCHOLMODTriplet =
        make_cholmod_wrapper(cholmod_allocate_triplet(tNumberOfRows, tNumberOfRows, tValues.size(),
                                                      tCHOLMODSTypeLowerDiagonal, CHOLMOD_REAL, &aCHOLMODCommon.mValue),
                             std::ref(aCHOLMODCommon));

    auto tCHOLMODCounter = CHOLMODIndexType{0};
    for (auto tRowIndex = 0; tRowIndex < tNumberOfRows; ++tRowIndex)
    {
        const auto tStartIndex = tRowEntrySpans[tRowIndex];
        const auto tEndIndex = tRowEntrySpans[tRowIndex + 1];
        for (auto tIndexIntoEntries = tStartIndex; tIndexIntoEntries < tEndIndex; ++tIndexIntoEntries)
        {
            const auto tColIndex = tColumns[tIndexIntoEntries];
            if (tColIndex <= tRowIndex)
            {
                static_cast<CHOLMODIndexType *>(tCHOLMODTriplet.mObject->i)[tCHOLMODCounter] = tRowIndex;
                static_cast<CHOLMODIndexType *>(tCHOLMODTriplet.mObject->j)[tCHOLMODCounter] = tColIndex;
                static_cast<double *>(tCHOLMODTriplet.mObject->x)[tCHOLMODCounter] = tValues[tIndexIntoEntries];
                ++tCHOLMODCounter;
            }
        }
    }

    tCHOLMODTriplet.mObject->nnz = tCHOLMODCounter;
    return make_cholmod_wrapper(
        cholmod_triplet_to_sparse(tCHOLMODTriplet.mObject, tCHOLMODCounter, &aCHOLMODCommon.mValue),
        std::ref(aCHOLMODCommon));
}

auto bad_cholmod_matrix_file_path() -> std::filesystem::path { return std::filesystem::path{"bad_cholmod_matrix.m"}; }
}  // namespace Plato::alg
