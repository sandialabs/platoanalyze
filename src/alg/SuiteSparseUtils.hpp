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

/// @brief Converts a CSR matrix to an equivalent matrix in CSC format.
///
/// UMFPACK and CHOLMOD use CSC format, but plato uses CSR.
[[nodiscard]] auto to_CSC(const CSRMatrix& aA) -> CSCMatrix;

/// @brief Converts a plato CrsMatrix that may have a block form. Copies the data to `std::vector`s.
[[nodiscard]] auto make_CSR_matrix(const Plato::CrsMatrix<Plato::OrdinalType>& aA) -> CSRMatrix;

/// @brief Converts a plato CrsMatrix that may have a block form. Copies the data to `std::vector`s.
///
/// This overload can be used with crs_matrix_non_block_form, so that properties of the matrix may be checked first,
/// such as symmetry.
[[nodiscard]] auto make_CSR_matrix(const CrsRowsColumnsValues<Plato::OrdinalType>& aRowsColumnsAndValues) -> CSRMatrix;

}  // namespace Plato::alg
