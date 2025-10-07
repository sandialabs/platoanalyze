#pragma once

#ifdef PLATO_UMFPACK

#include <umfpack.h>

#include <string>
#include <vector>

#include "PlatoAbstractSolver.hpp"
#include "PlatoStaticsTypes.hpp"
#include "alg/SuiteSparseUtils.hpp"

namespace Plato::alg
{

class UMFPACKLinearSolver : public Plato::AbstractSolver
{
   public:
    UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                        std::shared_ptr<Plato::MultipointConstraints> aMPCs = nullptr);

    void innerSolve(Plato::CrsMatrix<int> aA, Plato::ScalarVector aX, Plato::ScalarVector aB) override;
    void report_memory_usage();

   private:
    void check_umfpack(const std::string &msg);
    void clear();

    CSCMatrix mMatrix;
    std::array<double, UMFPACK_INFO> mInfo;
    void *mSymbolic = nullptr;
    void *mNumeric = nullptr;
};

}  // namespace Plato::alg

#endif
