#include <Teuchos_ParameterList.hpp>
#include <memory>
#include <string>

#include "core_types/PlatoTypes.hpp"
#include "domain/SpatialModel.hpp"
#include "parsing/ParseTools.hpp"
#include "solver/PlatoAbstractSolver.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "solver/multipoint_constraint/MultipointConstraints.hpp"
#include "solver/nonlinear_solvers/NewtonSolver.hpp"

namespace plato::parabolic
{
namespace
{
constexpr auto kMPCsName = std::string_view{"Multipoint Constraints"};
}
auto parse_multipoint_constraints(Teuchos::ParameterList& aProblemParams,
                                  const domain::SpatialModel& aSpatialModel,
                                  const Plato::OrdinalType aNumDofsPerNode)
    -> std::shared_ptr<Plato::MultipointConstraints>
{
    if (aProblemParams.isSublist(std::string{kMPCsName}) == true)
    {
        auto& tMPCsParams = aProblemParams.sublist(std::string{kMPCsName});
        const auto tMPCs = std::make_shared<Plato::MultipointConstraints>(aSpatialModel, aNumDofsPerNode, tMPCsParams);
        tMPCs->setupTransform();
        return tMPCs;
    }
    else
    {
        return nullptr;
    }
}

auto parse_linear_solver(Teuchos::ParameterList& aSolverParameters,
                         const std::string& aPhysics,
                         const Plato::OrdinalType aNumNodes,
                         Plato::Comm::Machine aMachine,
                         const Plato::OrdinalType aNumDofsPerNode,
                         const std::shared_ptr<Plato::MultipointConstraints>& aMPCs)
    -> Plato::rcp<Plato::AbstractSolver>
{
    const auto tSystemType = aPhysics == "Thermomechanical" ? Plato::LinearSystemType::SYMMETRIC_PATTERN
                                                            : Plato::LinearSystemType::SYMMETRIC_INDEFINITE;
    auto tSolverFactory = Plato::SolverFactory{aSolverParameters, tSystemType};
    return tSolverFactory.create(aNumNodes, aMachine, aNumDofsPerNode, aMPCs);
}

auto parse_newton_solver(Teuchos::ParameterList& aProblemParams, const Plato::rcp<Plato::AbstractSolver>& aLinearSolver)
    -> algorithms::nonlinear_solvers::NewtonSolver
{
    const auto tNumNewtonSteps = Plato::ParseTools::getSubParam<int>(aProblemParams, "Newton Iteration",
                                                                     "Maximum Iterations", /*aDefaultValue=*/1);
    const auto tNewtonResTol = Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration",
                                                                      "Residual Tolerance", /*aDefaultValue=*/0.0);
    const auto tNewtonIncTol = Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration",
                                                                      "Increment Tolerance", /*aDefaultValue=*/0.0);

    return algorithms::nonlinear_solvers::NewtonSolver{tNumNewtonSteps, tNewtonResTol, tNewtonIncTol, aLinearSolver};
}

}  // namespace plato::parabolic
