#include "alg/SuiteSparseSolverFactory.hpp"

#include "alg/CHOLMODLinearSolver.hpp"
#include "alg/UMFPACKLinearSolver.hpp"

namespace Plato::alg
{
auto make_suite_sparse_solver(const Teuchos::ParameterList& aSolverParams,
                              Plato::LinearSystemType aLinearSystemType,
                              std::shared_ptr<Plato::MultipointConstraints> aMPCs) -> std::unique_ptr<AbstractSolver>
{
    if (aLinearSystemType == LinearSystemType::SYMMETRIC_PATTERN)
    {
        return std::make_unique<UMFPACKLinearSolver>(aSolverParams, std::move(aMPCs));
    }
    return std::make_unique<CHOLMODLinearSolver>(aSolverParams, aLinearSystemType, std::move(aMPCs));
}
}  // namespace Plato::alg
