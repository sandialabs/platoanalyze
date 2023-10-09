#include <algorithm>
#include <array>
#include <math.h>

#include "alg/UMFPACKLinearSolver.hpp"

#include "util/PlatoTestHelpers.hpp"
#include "util/PlatoMathTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"

TEUCHOS_UNIT_TEST(UMFPACKSolver, Symmetric)
{
/*
     2    -1     0     0
    -1     2    -1     0
     0    -1     2    -1
     0     0    -1     2
*/

  namespace pu = Plato::UMFPACK;
  const pu::CSRMatrix Acsr = {/* .rowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                              /* .columns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                              /* .values = */ std::vector<double>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}};

  const pu::CSCMatrix Acsc = convertCSRtoCSC(Acsr);

  TEST_ASSERT(Acsc.colBegin == Acsr.rowBegin);
  TEST_ASSERT(Acsc.rows == Acsr.columns);
  TEST_ASSERT(Acsc.values == Acsr.values);
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, NonSymmetricEntries)
{
/*
     2     1     0     0
    -1     2     1     0
     0    -1     2     1
     0     0    -1     2
*/
  namespace pu = Plato::UMFPACK;
  const pu::CSRMatrix Acsr = {/* .rowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                              /* .columns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                              /* .values = */ std::vector<double>{2.0, 1.0, -1.0, 2.0, 1.0, -1.0, 2.0, 1.0, -1.0, 2.0}};

  const pu::CSCMatrix Acsc = convertCSRtoCSC(Acsr);

  const pu::CSCMatrix Acsc_gold = {/* .colBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                                   /* .rows = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                                   /* .values = */ std::vector<double>{2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0}};

  TEST_ASSERT(Acsc.colBegin == Acsc_gold.colBegin);
  TEST_ASSERT(Acsc.rows == Acsc_gold.rows);
  TEST_ASSERT(Acsc.values == Acsc_gold.values);
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, NonSymmetricSparsity)
{
/*
     1     3     2     0
     0     1     1     2
     0     4     3     2
     4     0     3     3
*/

  namespace pu = Plato::UMFPACK;
  const pu::CSRMatrix Acsr = {/* .rowBegin = */ std::vector<SuiteSparse_long>{0, 3, 6, 9, 12},
                              /* .columns = */ std::vector<SuiteSparse_long>{0, 1, 2, 1, 2, 3, 1, 2, 3, 0, 2, 3},
                              /* .values = */ std::vector<double>{1, 3, 2, 1, 1, 2, 4, 3, 2, 4, 3, 3}};

  const pu::CSCMatrix Acsc = convertCSRtoCSC(Acsr);

  const pu::CSCMatrix Acsc_gold = {/* .colBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 9, 12},
                                   /* .rows = */ std::vector<SuiteSparse_long>{0, 3, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3},
                                   /* .values = */ std::vector<double>{1, 4, 3, 1, 4, 2, 1, 3, 3, 2, 2, 3}};

  TEST_ASSERT(Acsc.colBegin == Acsc_gold.colBegin);
  TEST_ASSERT(Acsc.rows == Acsc_gold.rows);
  TEST_ASSERT(Acsc.values == Acsc_gold.values);
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, constructCSRMatrix)
{
/*
     2    -1     0     0
    -1     2    -1     0
     0    -1     2    -1
     0     0    -1     2
*/

  namespace pth = Plato::TestHelpers;

  const unsigned numRows = 4;
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(numRows, numRows, 1, 1) );
  std::vector<Plato::OrdinalType> tRowMapA = {0, 2, 5, 8, 10};
  std::vector<Plato::OrdinalType> tColMapA = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
  std::vector<Plato::Scalar>      tValuesA = {2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0};
  pth::set_matrix_data(tMatrixA, tRowMapA, tColMapA, tValuesA);

  namespace pu = Plato::UMFPACK;
  const pu::CSRMatrix A = pu::constructCSRMatrix(*tMatrixA);

  const pu::CSRMatrix Acsr_gold = {/* .rowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                                   /* .columns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                                   /* .values = */ std::vector<double>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}};

  TEST_ASSERT(A.rowBegin == Acsr_gold.rowBegin);
  TEST_ASSERT(A.columns == Acsr_gold.columns);
  TEST_ASSERT(A.values == Acsr_gold.values);
}
