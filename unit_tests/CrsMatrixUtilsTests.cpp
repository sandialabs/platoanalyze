#include <alg/CrsMatrixUtils.hpp>

#include <Teuchos_UnitTestHarness.hpp>

#include <array>

namespace
{
template<typename T, int ArraySize>
Kokkos::View<T*, Plato::MemSpace> deviceView(std::array<T, ArraySize> aArray)
{
    using DeviceView = Kokkos::View<T*, Plato::MemSpace>;
    using HostView = typename DeviceView::HostMirror;

    const HostView tArrayOnHost(aArray.data(), ArraySize);
    const DeviceView tArrayOnDevice("Array", ArraySize);
    Kokkos::deep_copy(tArrayOnDevice, tArrayOnHost);
    return tArrayOnDevice;
}
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternSymmetric)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 10;
    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector = deviceView<int, tNumRows + 1>({0, 2, 5, 8, tNumValues});
    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector = deviceView<int, tNumValues>({0, 1, 0, 1, 2, 1, 2, 3, 2, 3});
    typename Plato::CrsMatrix<int>::ScalarVectorT tValuesAsVector = deviceView<double, tNumValues>({2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0});
    Plato::CrsMatrix<int> tMatrix{tRowBeginAsVector, tColumnsAsVector, tValuesAsVector, tNumRows, tNumRows, 1, 1};
    TEST_ASSERT(Plato::has_symmetric_sparsity_pattern(tMatrix));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternSymmetricFull)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 16;
    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector = deviceView<int, tNumRows + 1>({0, 4, 8, 12, tNumValues});
    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector = deviceView<int, tNumValues>({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
    typename Plato::CrsMatrix<int>::ScalarVectorT tValuesAsVector = deviceView<double, tNumValues>({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
    Plato::CrsMatrix<int> tMatrix{tRowBeginAsVector, tColumnsAsVector, tValuesAsVector, tNumRows, tNumRows, 1, 1};
    TEST_ASSERT(Plato::has_symmetric_sparsity_pattern(tMatrix));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternNonsymmetricUpperTri)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 11;
    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector = deviceView<int, tNumRows + 1>({0, 3, 5, 8, tNumValues});
    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector = deviceView<int, tNumValues>({0, 1, 2, 0, 1, 2, 1, 2, 3, 2, 3});
    typename Plato::CrsMatrix<int>::ScalarVectorT tValuesAsVector = deviceView<double, tNumValues>({2.0, -1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0});
    Plato::CrsMatrix<int> tMatrix{tRowBeginAsVector, tColumnsAsVector, tValuesAsVector, tNumRows, tNumRows, 1, 1};
    TEST_ASSERT(!Plato::has_symmetric_sparsity_pattern(tMatrix));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternNonsymmetricLowerTri)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 11;
    typename Plato::CrsMatrix<int>::OrdinalVectorT tRowBeginAsVector = deviceView<int, tNumRows + 1>({0, 2, 5, 9, tNumValues});
    typename Plato::CrsMatrix<int>::RowMapVectorT tColumnsAsVector = deviceView<int, tNumValues>({0, 1, 0, 1, 2, 0, 1, 2, 3, 2, 3});
    typename Plato::CrsMatrix<int>::ScalarVectorT tValuesAsVector = deviceView<double, tNumValues>({2.0, -1.0, -1.0, 2.0, -1.0, 1.0, -1.0, 2.0, -1.0, -1.0, 2.0});
    Plato::CrsMatrix<int> tMatrix{tRowBeginAsVector, tColumnsAsVector, tValuesAsVector, tNumRows, tNumRows, 1, 1};
    TEST_ASSERT(!Plato::has_symmetric_sparsity_pattern(tMatrix));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, SparsityPatternHash)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 11;

    typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector = deviceView<int, tNumRows + 1>({0, 2, 5, 9, tNumValues});
    typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector = deviceView<int, tNumValues>({0, 1, 0, 1, 2, 0, 1, 2, 3, 2, 3});

    const std::size_t tHash1 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector, tColumnsAsVector);
    {
        const std::size_t tHash2 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector, tColumnsAsVector);
        TEST_EQUALITY(tHash1, tHash2);
    }

    // Different column index
    {
        typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector2 = deviceView<int, tNumValues>({0, 2, 0, 1, 2, 0, 1, 2, 3, 2, 3});
        const std::size_t tHash2 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector, tColumnsAsVector2);
        TEST_INEQUALITY(tHash1, tHash2)
    }
    // Different row counts
    {
        typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector2 = deviceView<int, tNumRows + 1>({0, 3, 5, 9, tNumValues});
        const std::size_t tHash2 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector2, tColumnsAsVector);
        TEST_INEQUALITY(tHash1, tHash2)
    }
    // Different total number of entries
    {
        constexpr int tNumValues2 = 16;
        typename Plato::CrsMatrix<int>::RowMapVectorT tRowBeginAsVector2 = deviceView<int, tNumRows + 1>({0, 4, 8, 12, tNumValues2});
        typename Plato::CrsMatrix<int>::OrdinalVectorT tColumnsAsVector2 = deviceView<int, tNumValues2>({0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3});
        const std::size_t tHash2 = Plato::crs_matrix_row_column_hash<int>(tRowBeginAsVector2, tColumnsAsVector2);
        TEST_INEQUALITY(tHash1, tHash2)
    }
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, ValidCRSMatrixMaps_AllZeroSizes)
{
    Plato::OrdinalVector tRows;
    Plato::OrdinalVector tCols;
    Plato::ScalarVector tEntries;
    TEST_ASSERT(Plato::validCrsMapSizes(tRows, tCols, tEntries));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, ValidCRSMatrixMaps_ValidNonBlock)
{
    constexpr int tNumRows = 4;
    constexpr int tNumValues = 11;

    const Plato::OrdinalVector tRowBeginDevice = deviceView<int, tNumRows + 1>({0, 2, 5, 9, tNumValues});
    const Plato::OrdinalVector tColsDevice = deviceView<int, tNumValues>({0, 1, 0, 1, 2, 0, 1, 2, 3, 2, 3});
    const Plato::ScalarVector tEntriesDevice = deviceView<double, tNumValues>({0, 1, 2, 3, 4, 5, 6, 7, 8 , 9, 10});

    // Test valid CRS matrix
    TEST_ASSERT(Plato::validCrsMapSizes(tRowBeginDevice, tColsDevice, tEntriesDevice));
    // Test invalid block using same matrix
    {
        constexpr int tNumRowsPerBlock = 2;
        constexpr int tNumColsPerBlock = 1;
        TEST_ASSERT(!Plato::validCrsMapSizes(tRowBeginDevice, tColsDevice, tEntriesDevice, tNumRowsPerBlock, tNumColsPerBlock));
    }
    // Test invalid block using same matrix
    {
        constexpr int tNumRowsPerBlock = 1;
        constexpr int tNumColsPerBlock = 2;
        TEST_ASSERT(!Plato::validCrsMapSizes(tRowBeginDevice, tColsDevice, tEntriesDevice, tNumRowsPerBlock, tNumColsPerBlock));
    }
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, ValidCRSMatrixMaps_ValidBlock)
{
    constexpr int tNumRows = 2;
    constexpr int tNumValues = 4;
    constexpr int tNumRowsPerBlock = 2;
    constexpr int tNumColsPerBlock = 1;

    const Plato::OrdinalVector tRowBeginDevice = deviceView<int, tNumRows + 1>({0, 2, tNumValues});
    const Plato::OrdinalVector tColsDevice = deviceView<int, tNumValues>({0, 2, 1, 3});
    const Plato::ScalarVector tEntriesDevice = deviceView<double, tNumValues * tNumRowsPerBlock * tNumColsPerBlock>({0, 1, 2, 3, 4, 5, 6, 7});

    // Valid
    TEST_ASSERT(Plato::validCrsMapSizes(tRowBeginDevice, tColsDevice, tEntriesDevice, tNumRowsPerBlock, tNumColsPerBlock));
    // Test invalid non-block using same matrix, which will have the wrong number of entries
    TEST_ASSERT(!Plato::validCrsMapSizes(tRowBeginDevice, tColsDevice, tEntriesDevice));
}

TEUCHOS_UNIT_TEST(CrsMatrixUtils, ValidCRSMatrixMaps_InvalidNumberOfColumns)
{
    constexpr int tNumRows = 1;
    constexpr int tNumExpectedValues = 1;
    constexpr int tNumErroneousValues = 4;

    const Plato::OrdinalVector tRowBeginDevice = deviceView<int, tNumRows + 1>({0, tNumExpectedValues});
    const Plato::OrdinalVector tColsDevice = deviceView<int, tNumErroneousValues>({0, 2, 4, 6});
    const Plato::ScalarVector tEntriesDevice = deviceView<double, tNumExpectedValues>({0});

    TEST_ASSERT(!Plato::validCrsMapSizes(tRowBeginDevice, tColsDevice, tEntriesDevice));
}
