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

auto CRSMatrix::numberOfRows() const -> SuiteSparse_long { return mRowBegin.size() - 1; }

auto CCSMatrix::numberOfColumns() const -> SuiteSparse_long { return mColumnBegin.size() - 1; }

CRSMatrix make_CRS_matrix(const Plato::CrsMatrix<Plato::OrdinalType>& aA)
{
    return make_CRS_matrix(Plato::crs_matrix_non_block_form<Plato::OrdinalType>(aA));
}

auto make_CRS_matrix(const CrsRowsColumnsValues<Plato::OrdinalType>& aRowsColumnsAndValues) -> CRSMatrix
{
    const auto& [tRowBegin, tColumns, tValues] = aRowsColumnsAndValues;
    return CRSMatrix{kokkos_view_to_std_vector<SuiteSparse_long>(tRowBegin),
                     kokkos_view_to_std_vector<SuiteSparse_long>(tColumns), kokkos_view_to_std_vector<double>(tValues)};
}

CCSMatrix to_CCS(const CRSMatrix& aMatrix)
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

    auto tCCSMatrix = CCSMatrix{};
    tCCSMatrix.mColumnBegin.resize(tNumberOfRows + 1);
    tCCSMatrix.mRows.resize(tNumberOfEntries);
    tCCSMatrix.mValues.resize(tNumberOfEntries);

    if (UMFPACK_OK != umfpack_dl_triplet_to_col(tNumberOfRows, tNumberOfRows, tNumberOfEntries, rows.data(),
                                                aMatrix.mColumns.data(), aMatrix.mValues.data(),
                                                tCCSMatrix.mColumnBegin.data(), tCCSMatrix.mRows.data(),
                                                tCCSMatrix.mValues.data(), nullptr))
    {
        ANALYZE_THROWERR("Triplet to column conversion failed.");
    }

    return tCCSMatrix;
}

}  // namespace Plato::alg
