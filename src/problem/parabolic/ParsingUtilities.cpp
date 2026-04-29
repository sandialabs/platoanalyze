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
#include "utilities/AnalyzeMacros.hpp"

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

void check_time_step(Teuchos::ParameterList& aProblemParams, const plato::domain::SpatialDomain& aSpatialDomain)
{
    auto tModelsParamList = aProblemParams.get<Teuchos::ParameterList>("Material Models");
    auto tModelParamList = tModelsParamList.sublist(aSpatialDomain.materialName());
    auto tConductionParameters = tModelParamList.sublist("Thermal Conduction");
    auto tThermalMassParameters = tModelParamList.sublist("Thermal Mass");
    const auto tConductivity = tConductionParameters.get<Plato::Scalar>("Thermal Conductivity");
    const auto tDensity = tThermalMassParameters.get<Plato::Scalar>("Mass Density");
    const auto tSpecificHeat = tThermalMassParameters.get<Plato::Scalar>("Specific Heat");
    const auto tDiffusivity = tConductivity / tDensity / tSpecificHeat;
    const auto tTimeStep =
        Plato::ParseTools::getSubParam<Plato::Scalar>(aProblemParams, "Time Integration", "Time Step", 1.0);

    const auto tFourierLimit = std::sqrt(2.0 * tDiffusivity * tTimeStep);

    const std::string tErrorMessage =
        std::string(
            "To have a Fourier number less than 0.5 for the provided material properties "
            "and time step, the characteristic mesh size should be greater than '") +
        std::to_string(tFourierLimit) +
        "'. If the characteristic mesh size is below this, the time step should be decreased to ensure accuracy.";
    WARNING(tErrorMessage)
}

}  // namespace plato::parabolic
