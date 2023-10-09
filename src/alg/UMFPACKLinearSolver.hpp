#pragma once

#include <vector>

#include "PlatoAbstractSolver.hpp"
#include "PlatoStaticsTypes.hpp"

#include <umfpack.h>

namespace Plato::UMFPACK {

struct CSRMatrix {
    std::vector<SuiteSparse_long> rowBegin;
    std::vector<SuiteSparse_long> columns;
    std::vector<double> values;
};

struct CSCMatrix {
    std::vector<SuiteSparse_long> colBegin;
    std::vector<SuiteSparse_long> rows;
    std::vector<double> values;
};

CSCMatrix convertCSRtoCSC(const CSRMatrix &A);

class UMFPACKLinearSolver : public Plato::AbstractSolver
{
public:
    UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                        std::shared_ptr<Plato::MultipointConstraints> aMPCs = nullptr);
    virtual ~UMFPACKLinearSolver();

    void innerSolve(
        Plato::CrsMatrix<int> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) override;
    void report_memory_usage();
private:
    void check_umfpack(const char *msg);
    uint64_t ne;
    std::vector<SuiteSparse_long> Ap, Ai;
    std::vector<double> Ax;
    std::array<double,UMFPACK_INFO> Info;
    void *Symbolic = nullptr, *Numeric = nullptr;
};

} // namespace UMFPACK