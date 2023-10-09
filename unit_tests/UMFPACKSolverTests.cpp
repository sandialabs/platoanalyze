#include <algorithm>
#include <array>
#include <math.h>

#include "alg/UMFPACKLinearSolver.hpp"

#include "Teuchos_UnitTestHarness.hpp"

TEUCHOS_UNIT_TEST(UMFPACKSolver, Symmetric)
{
  namespace pu = Plato::UMFPACK;
  const pu::CSRMatrix Acsr = {/* .rowBegin = */ std::vector<SuiteSparse_long>{0, 2, 5, 8, 10},
                              /* .columns = */ std::vector<SuiteSparse_long>{0, 1, 0, 1, 2, 1, 2, 3, 2, 3},
                              /* .values = */ std::vector<double>{2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0}};

  const pu::CSCMatrix Acsc = convertCSRtoCSC(Acsr);

  TEST_ASSERT(Acsc.colBegin == Acsr.rowBegin);
  TEST_ASSERT(Acsc.rows == Acsr.columns);
  TEST_ASSERT(Acsc.values == Acsr.values);
}

TEUCHOS_UNIT_TEST(UMFPACKSolver, NonSymmetric)
{
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
