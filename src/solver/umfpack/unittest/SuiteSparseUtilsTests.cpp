#include <math.h>

#include <Teuchos_UnitTestHarness.hpp>
#include <algorithm>
#include <array>

#include "linear_algebra/CrsMatrixUtils.hpp"
#include "solver/umfpack/SuiteSparseUtils.hpp"
#include "test_utilities/PlatoMathTestHelpers.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

TEUCHOS_UNIT_TEST(SuiteSparseUtils, SymmetricCCSConversion)
{
    /*
         2    -1     0     0
        -1     2    -1     0
         0    -1     2    -1
         0     0    -1     2
    */

    namespace pa = Plato::alg;
    const auto tAAsCRS =
        pa::CRSMatrix{/* .mRowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                      /* .mColumns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                      /* .mValues = */ std::vector<double>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}};

    const auto tAAsCCS = to_CCS(tAAsCRS);

    TEST_ASSERT(tAAsCCS.mColumnBegin == tAAsCRS.mRowBegin);
    TEST_ASSERT(tAAsCCS.mRows == tAAsCRS.mColumns);
    TEST_ASSERT(tAAsCCS.mValues == tAAsCRS.mValues);
}

TEUCHOS_UNIT_TEST(SuiteSparseUtils, NonSymmetricEntriesCCSConversion)
{
    /*
         2     1     0     0
        -1     2     1     0
         0    -1     2     1
         0     0    -1     2
    */
    namespace pa = Plato::alg;
    const auto tAAsCRS =
        pa::CRSMatrix{/* .mRowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                      /* .mColumns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                      /* .mValues = */ std::vector<double>{2.0, 1.0, -1.0, 2.0, 1.0, -1.0, 2.0, 1.0, -1.0, 2.0}};

    const pa::CCSMatrix tAAsCCS = to_CCS(tAAsCRS);

    const auto tAAsCCSExpected =
        pa::CCSMatrix{/* .mColumnBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                      /* .mRows = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                      /* .mValues = */ std::vector<double>{2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0}};

    TEST_ASSERT(tAAsCCS.mColumnBegin == tAAsCCSExpected.mColumnBegin);
    TEST_ASSERT(tAAsCCS.mRows == tAAsCCSExpected.mRows);
    TEST_ASSERT(tAAsCCS.mValues == tAAsCCSExpected.mValues);
}

TEUCHOS_UNIT_TEST(SuiteSparseUtils, NonSymmetricSparsityCCSConversion)
{
    /*
         1     3     2     0
         0     1     1     2
         0     4     3     2
         4     0     3     3
    */

    namespace pa = Plato::alg;
    const auto tAAsCRS =
        pa::CRSMatrix{/* .mRowBegin = */ std::vector<SuiteSparse_long>{0, 3, 6, 9, 12},
                      /* .mColumns = */ std::vector<SuiteSparse_long>{0, 1, 2, 1, 2, 3, 1, 2, 3, 0, 2, 3},
                      /* .mValues = */ std::vector<double>{1, 3, 2, 1, 1, 2, 4, 3, 2, 4, 3, 3}};

    const auto tAAsCCS = to_CCS(tAAsCRS);

    const auto tAAsCCSExpected =
        pa::CCSMatrix{/* .mColumnBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 9, 12},
                      /* .mRows = */ std::vector<SuiteSparse_long>{0, 3, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3},
                      /* .mValues = */ std::vector<double>{1, 4, 3, 1, 4, 2, 1, 3, 3, 2, 2, 3}};

    TEST_ASSERT(tAAsCCS.mColumnBegin == tAAsCCSExpected.mColumnBegin);
    TEST_ASSERT(tAAsCCS.mRows == tAAsCCSExpected.mRows);
    TEST_ASSERT(tAAsCCS.mValues == tAAsCCSExpected.mValues);
}

TEUCHOS_UNIT_TEST(SuiteSparseUtils, MakeCRSMatrix)
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
    const pa::CRSMatrix tAAsCRSExpected = {
        /* .mRowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
        /* .mColumns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
        /* .mValues = */ std::vector<double>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}};

    // Direct from CrsMatrix
    {
        const auto tA = pa::make_CRS_matrix(*tMatrixA);
        TEST_ASSERT(tA.mRowBegin == tAAsCRSExpected.mRowBegin);
        TEST_ASSERT(tA.mColumns == tAAsCRSExpected.mColumns);
        TEST_ASSERT(tA.mValues == tAAsCRSExpected.mValues);
    }
    // Overload
    {
        const auto tA = pa::make_CRS_matrix(Plato::crs_matrix_non_block_form<Plato::OrdinalType>(*tMatrixA));
        TEST_ASSERT(tA.mRowBegin == tAAsCRSExpected.mRowBegin);
        TEST_ASSERT(tA.mColumns == tAAsCRSExpected.mColumns);
        TEST_ASSERT(tA.mValues == tAAsCRSExpected.mValues);
    }
}
