#pragma once

#ifdef PLATO_UMFPACK

#include <vector>
#include <string>

#include "PlatoAbstractSolver.hpp"
#include "PlatoStaticsTypes.hpp"

#include <umfpack.h>

namespace Plato::UMFPACK {

struct CSRMatrix {
    std::vector<SuiteSparse_long> rowBegin;
    std::vector<SuiteSparse_long> columns;
    std::vector<double> values;
    SuiteSparse_long nRows() const { return rowBegin.size()-1; }
};

struct CSCMatrix {
    std::vector<SuiteSparse_long> colBegin;
    std::vector<SuiteSparse_long> rows;
    std::vector<double> values;
    SuiteSparse_long nCols() const { return colBegin.size()-1; }
};

CSCMatrix convertCSRtoCSC(const CSRMatrix &A);
CSRMatrix constructCSRMatrix(const Plato::CrsMatrix<int> &aA);

class UMFPACKLinearSolver : public Plato::AbstractSolver
{
public:
    UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                        std::shared_ptr<Plato::MultipointConstraints> aMPCs = nullptr);

    void innerSolve(
        Plato::CrsMatrix<int> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) override;
    void report_memory_usage();
private:
    void check_umfpack(const std::string &msg);
    void clear();
    CSCMatrix mMatrix;
    std::array<double,UMFPACK_INFO> mInfo;
    void *mSymbolic = nullptr;
    void *mNumeric = nullptr;
};

} // namespace UMFPACK

#endif
