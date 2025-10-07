#pragma once

#include <SuiteSparse_config.h>

#include <vector>

#include "CrsMatrix.hpp"

namespace Plato::alg
{
/// @brief Representation of a CSR matrix using `std::vector` data types.
///
/// The purpose of this class is for converting Plato::CSRMatrix to C-style arrays used in UMFPACK and CHOLMOD.
struct CSRMatrix
{
    std::vector<SuiteSparse_long> mRowBegin;
    std::vector<SuiteSparse_long> mColumns;
    std::vector<double> mValues;
    [[nodiscard]] auto numberOfRows() const -> SuiteSparse_long;
};

/// @brief Representation of a CSC matrix using `std::vector` data types.
///
/// The purpose of this class is for converting Plato::CSRMatrix to C-style arrays used in UMFPACK and CHOLMOD.
struct CSCMatrix
{
    std::vector<SuiteSparse_long> mColumnBegin;
    std::vector<SuiteSparse_long> mRows;
    std::vector<double> mValues;
    [[nodiscard]] auto numberOfColumns() const -> SuiteSparse_long;
};

[[nodiscard]] auto convertCSRtoCSC(const CSRMatrix &aA) -> CSCMatrix;
[[nodiscard]] auto constructCSRMatrix(const Plato::CrsMatrix<int> &aA) -> CSRMatrix;

}  // namespace Plato::alg
