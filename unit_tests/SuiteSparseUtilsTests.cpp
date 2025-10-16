#include <math.h>

#include <algorithm>
#include <array>

#include "Teuchos_UnitTestHarness.hpp"
#include "alg/CrsMatrixUtils.hpp"
#include "alg/SuiteSparseUtils.hpp"
#include "util/PlatoMathTestHelpers.hpp"
#include "util/PlatoTestHelpers.hpp"

TEUCHOS_UNIT_TEST(UMFPACKSolver, Symmetric)
{
    /*
         2    -1     0     0
        -1     2    -1     0
         0    -1     2    -1
         0     0    -1     2
    */

    namespace pa = Plato::alg;
    const auto tAAsCSR =
        pa::CSRMatrix{/* .mRowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                      /* .mColumns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                      /* .mValues = */ std::vector<double>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}};

    const auto tAAsCSC = to_CSC(tAAsCSR);

    TEST_ASSERT(tAAsCSC.mColumnBegin == tAAsCSR.mRowBegin);
    TEST_ASSERT(tAAsCSC.mRows == tAAsCSR.mColumns);
    TEST_ASSERT(tAAsCSC.mValues == tAAsCSR.mValues);
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, NonSymmetricEntries)
{
    /*
         2     1     0     0
        -1     2     1     0
         0    -1     2     1
         0     0    -1     2
    */
    namespace pa = Plato::alg;
    const pa::CSRMatrix tAAsCSR = {
        /* .mRowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
        /* .mColumns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
        /* .mValues = */ std::vector<double>{2.0, 1.0, -1.0, 2.0, 1.0, -1.0, 2.0, 1.0, -1.0, 2.0}};

    const pa::CSCMatrix tAAsCSC = to_CSC(tAAsCSR);

    const pa::CSCMatrix tAAsCSCExpected = {
        /* .mColumnBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
        /* .mRows = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
        /* .mValues = */ std::vector<double>{2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0}};

    TEST_ASSERT(tAAsCSC.mColumnBegin == tAAsCSCExpected.mColumnBegin);
    TEST_ASSERT(tAAsCSC.mRows == tAAsCSCExpected.mRows);
    TEST_ASSERT(tAAsCSC.mValues == tAAsCSCExpected.mValues);
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, NonSymmetricSparsity)
{
    /*
         1     3     2     0
         0     1     1     2
         0     4     3     2
         4     0     3     3
    */

    namespace pa = Plato::alg;
    const auto tAAsCSR =
        pa::CSRMatrix{/* .mRowBegin = */ std::vector<SuiteSparse_long>{0, 3, 6, 9, 12},
                      /* .mColumns = */ std::vector<SuiteSparse_long>{0, 1, 2, 1, 2, 3, 1, 2, 3, 0, 2, 3},
                      /* .mValues = */ std::vector<double>{1, 3, 2, 1, 1, 2, 4, 3, 2, 4, 3, 3}};

    const auto tAAsCSC = to_CSC(tAAsCSR);

    const auto tAAsCSCExpected =
        pa::CSCMatrix{/* .mColumnBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 9, 12},
                      /* .mRows = */ std::vector<SuiteSparse_long>{0, 3, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3},
                      /* .mValues = */ std::vector<double>{1, 4, 3, 1, 4, 2, 1, 3, 3, 2, 2, 3}};

    TEST_ASSERT(tAAsCSC.mColumnBegin == tAAsCSCExpected.mColumnBegin);
    TEST_ASSERT(tAAsCSC.mRows == tAAsCSCExpected.mRows);
    TEST_ASSERT(tAAsCSC.mValues == tAAsCSCExpected.mValues);
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, MakeCSRMatrix)
{
    /*
         2    -1     0     0
        -1     2    -1     0
         0    -1     2    -1
         0     0    -1     2
    */

    namespace pth = Plato::TestHelpers;

    const unsigned tNumberOfRows = 4;
    auto tMatrixA = Teuchos::rcp(new Plato::CrsMatrixType(tNumberOfRows, tNumberOfRows, 1, 1));
    std::vector<Plato::OrdinalType> tRowMapA = {0, 2, 5, 8, 10};
    std::vector<Plato::OrdinalType> tColMapA = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    std::vector<Plato::Scalar> tValuesA = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
    pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

    namespace pa = Plato::alg;
    const pa::CSRMatrix tAAsCSRExpected = {
        /* .mRowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
        /* .mColumns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
        /* .mValues = */ std::vector<double>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}};

    // Direct from CrsMatrix
    {
        const auto tA = pa::make_CSR_matrix(*tMatrixA);
        TEST_ASSERT(tA.mRowBegin == tAAsCSRExpected.mRowBegin);
        TEST_ASSERT(tA.mColumns == tAAsCSRExpected.mColumns);
        TEST_ASSERT(tA.mValues == tAAsCSRExpected.mValues);
    }
    // Overload
    {
        const auto tA = pa::make_CSR_matrix(Plato::crs_matrix_non_block_form<Plato::OrdinalType>(*tMatrixA));
        TEST_ASSERT(tA.mRowBegin == tAAsCSRExpected.mRowBegin);
        TEST_ASSERT(tA.mColumns == tAAsCSRExpected.mColumns);
        TEST_ASSERT(tA.mValues == tAAsCSRExpected.mValues);
    }
}
