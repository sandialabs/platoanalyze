#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <memory>

#include "solver/PlatoAbstractSolver.hpp"
#include "solver/umfpack/CHOLMODLinearSolver.hpp"
#include "solver/umfpack/SuiteSparseSolverFactory.hpp"
#include "solver/umfpack/UMFPACKLinearSolver.hpp"

namespace
{
template <typename DerivedSolver>
[[nodiscard]] auto check_solver_type(const Plato::LinearSystemType aLinearSystemType) -> bool
{
    const auto tSolver = std::shared_ptr<Plato::AbstractSolver>{
        Plato::alg::make_suite_sparse_solver(Teuchos::ParameterList{}, aLinearSystemType)};
    const auto tSolverAsDerived = std::dynamic_pointer_cast<DerivedSolver>(tSolver);
    return static_cast<bool>(tSolverAsDerived);
}
}  // namespace

TEUCHOS_UNIT_TEST(SuiteSparseSolverFactory, CreatesCorrectType)
{
    TEST_ASSERT(
        check_solver_type<Plato::alg::CHOLMODLinearSolver>(Plato::LinearSystemType::SYMMETRIC_POSITIVE_DEFINITE));
    TEST_ASSERT(check_solver_type<Plato::alg::CHOLMODLinearSolver>(Plato::LinearSystemType::SYMMETRIC_INDEFINITE));
    TEST_ASSERT(check_solver_type<Plato::alg::UMFPACKLinearSolver>(Plato::LinearSystemType::SYMMETRIC_PATTERN));
}
