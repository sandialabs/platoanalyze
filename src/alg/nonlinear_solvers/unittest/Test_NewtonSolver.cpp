#include <mpi.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <cmath>

#include "PlatoMathTestHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTestHelpers.hpp"
#include "PlatoTypes.hpp"
#include "alg/CrsMatrix.hpp"
#include "alg/ParallelComm.hpp"
#include "alg/PlatoAbstractSolver.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "alg/nonlinear_solvers/NewtonSolver.hpp"

namespace plato::algorithms::nonlinear_solvers::unittest
{
namespace
{
/// @brief 2D Nonlinear system of equations for testing Newton method.
/// From: Nocedal, J., & Wright, S. J. (Eds.). (1999). Numerical optimization. New York, NY: Springer New York.
/// This system has a root at [0, 1]
struct TestNonlinearSystem
{
    [[nodiscard]] Plato::ScalarVector residual(const Plato::ScalarVector& aState)
    {
        const auto tHostState = Plato::TestHelpers::get(aState);
        const Plato::Scalar tX = tHostState(0);
        const Plato::Scalar tY = tHostState(1);

        std::vector<Plato::Scalar> tResidualVec{(tX + 3.0) * (tY * tY * tY - 7.0) + 18.0,
                                                std::sin(tY * std::exp(tX) - 1.0)};

        return Plato::TestHelpers::create_device_view(tResidualVec);
    }

    [[nodiscard]] Teuchos::RCP<Plato::CrsMatrixType> jacobian(const Plato::ScalarVector& aState)
    {
        const auto tHostState = Plato::TestHelpers::get(aState);
        const Plato::Scalar tX = tHostState(0);
        const Plato::Scalar tY = tHostState(1);

        const auto tMatrix = Teuchos::rcp(new Plato::CrsMatrixType(2, 2, 1, 1));
        const std::vector<Plato::OrdinalType> tRowMap = {0, 2, 4};
        const std::vector<Plato::OrdinalType> tColMap = {0, 1, 0, 1};
        const std::vector<Plato::Scalar> tValues = {tY * tY * tY - 7.0, 3.0 * tY * tY * (tX + 3.0),
                                                    std::cos(tY * std::exp(tX) - 1.0) * tY * std::exp(tX),
                                                    std::cos(tY * std::exp(tX) - 1.0) * std::exp(tX)};
        Plato::TestHelpers::set_matrix_data(tMatrix, tRowMap, tColMap, tValues);
        return tMatrix;
    }
};
}  // namespace

TEUCHOS_UNIT_TEST(NewtonSolver, FindsExpectedRoot)
{
    auto tNonlinearSystem = TestNonlinearSystem{};

    auto computeResidual = [&tNonlinearSystem](const Plato::ScalarVector& aState) -> Plato::ScalarVector
    { return tNonlinearSystem.residual(aState); };

    auto computeJacobian = [&tNonlinearSystem](const Plato::ScalarVector& aState) -> Teuchos::RCP<Plato::CrsMatrixType>
    { return tNonlinearSystem.jacobian(aState); };

    auto applyBoundaryConditions = [](Teuchos::RCP<Plato::CrsMatrixType>& aMatrix, Plato::ScalarVector& aVector,
                                      const Plato::Scalar aScale) {};

    auto tSolverParams = Teuchos::ParameterList{};
    tSolverParams.set("Solver Stack", "Tpetra");  // Jacobian is not symmetric
    Plato::SolverFactory tSolverFactory(tSolverParams);
    constexpr Plato::OrdinalType tDofsPerNode{1};
    constexpr Plato::OrdinalType tNumNodes{2};
    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);
    Plato::rcp<Plato::AbstractSolver> tSolver = tSolverFactory.create(tNumNodes, tMachine, tDofsPerNode);

    constexpr Plato::OrdinalType tNewtonSteps{20};
    constexpr Plato::Scalar tResidualTol{1e-12};
    constexpr Plato::Scalar tIncrementTol{1e-16};

    const auto tNewtonSolver = NewtonSolver{tNewtonSteps, tResidualTol, tIncrementTol, tSolver};

    std::vector<Plato::Scalar> tInitialPoint{-0.5, 1.4};
    auto tState = Plato::TestHelpers::create_device_view(tInitialPoint);

    const bool tNewtonHasConverged =
        tNewtonSolver.solve(tState, computeResidual, computeJacobian, applyBoundaryConditions);
    TEST_ASSERT(tNewtonHasConverged);

    const auto tSolutionHost = Plato::TestHelpers::get(tState);
    constexpr Plato::Scalar tTestTol = 1e-14;
    TEST_ASSERT(std::abs(tSolutionHost(0)) <
                tTestTol);  // can't test with TEST_FLOATING_EQUALITY because of 0 when computing relative difference
    TEST_FLOATING_EQUALITY(tSolutionHost(1), 1.0, tTestTol);
}
}  // namespace plato::algorithms::nonlinear_solvers::unittest
