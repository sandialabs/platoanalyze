#include "alg/SuiteSparseUtils.hpp"

#include <Kokkos_StdAlgorithms.hpp>

#include "CrsMatrixUtils.hpp"
#include "umfpack.h"

namespace Plato::alg
{
namespace
{
template <typename ReturnType, typename ViewType>
[[nodiscard]] auto kokkos_view_to_std_vector(ViewType aView) -> std::vector<ReturnType>
{
    static_assert(ViewType::rank() == 1, "invalid usage of kokkos_view_to_std_vector: requires one dimension");

    const auto tMirror = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, aView);
    auto tVectorCopy = std::vector<ReturnType>{};
    tVectorCopy.reserve(aView.size());
    std::copy(Kokkos::Experimental::begin(tMirror), Kokkos::Experimental::end(tMirror),
              std::back_inserter(tVectorCopy));

    return tVectorCopy;
}
}  // namespace

auto CSRMatrix::numberOfRows() const -> SuiteSparse_long { return mRowBegin.size() - 1; }

auto CSCMatrix::numberOfColumns() const -> SuiteSparse_long { return mColumnBegin.size() - 1; }

CSRMatrix make_CSR_matrix(const Plato::CrsMatrix<int>& aA)
{
    using CrsOrdinal = int;
    const auto [tRowBegin, tColumns, tValues] = Plato::crs_matrix_non_block_form<CrsOrdinal>(aA);
    return make_CSR_matrix(tRowBegin, tColumns, tValues);
}

auto make_CSR_matrix(typename Plato::CrsMatrix<int>::RowMapVectorT aRowBegin,
                     typename Plato::CrsMatrix<int>::OrdinalVectorT aColumns,
                     typename Plato::CrsMatrix<int>::ScalarVectorT aValues) -> CSRMatrix
{
    return CSRMatrix{kokkos_view_to_std_vector<SuiteSparse_long>(aRowBegin),
                     kokkos_view_to_std_vector<SuiteSparse_long>(aColumns), kokkos_view_to_std_vector<double>(aValues)};
}

CSCMatrix to_CSC(const CSRMatrix& aMatrix)
{
    assert(aMatrix.mRowBegin.size() > 0);
    assert(aMatrix.mColumns.size() == aMatrix.mValues.size());
    const auto tNumberOfRows = aMatrix.numberOfRows();
    const auto tNumberOfEntries = aMatrix.mColumns.size();

    auto rows = std::vector<SuiteSparse_long>(tNumberOfEntries);

    if (UMFPACK_OK != umfpack_dl_col_to_triplet(tNumberOfRows, aMatrix.mRowBegin.data(), rows.data()))
    {
        ANALYZE_THROWERR("Column to triplet conversion failed.");
    }

    auto tCSCMatrix = CSCMatrix{};
    tCSCMatrix.mColumnBegin.resize(tNumberOfRows + 1);
    tCSCMatrix.mRows.resize(tNumberOfEntries);
    tCSCMatrix.mValues.resize(tNumberOfEntries);

    if (UMFPACK_OK != umfpack_dl_triplet_to_col(tNumberOfRows, tNumberOfRows, tNumberOfEntries, rows.data(),
                                                aMatrix.mColumns.data(), aMatrix.mValues.data(),
                                                tCSCMatrix.mColumnBegin.data(), tCSCMatrix.mRows.data(),
                                                tCSCMatrix.mValues.data(), nullptr))
    {
        ANALYZE_THROWERR("Triplet to column conversion failed.");
    }

    return tCSCMatrix;
}

}  // namespace Plato::alg
