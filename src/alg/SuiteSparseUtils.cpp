#include "alg/SuiteSparseUtils.hpp"

namespace Plato::alg
{
auto CSRMatrix::numberOfRows() const -> SuiteSparse_long { return mRowBegin.size() - 1; }

auto CSCMatrix::numberOfColumns() const -> SuiteSparse_long { return mColumnBegin.size() - 1; }
}  // namespace Plato::alg
