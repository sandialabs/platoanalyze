#include <mpi.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <plato/test_utilities/GradientChecker.hpp>
#include <string>

#include "element/Tri3.hpp"
#include "problem/Thermal.hpp"
#include "problem/parabolic/Problem.hpp"
#include "test_utilities/PlatoGradientCheckTestHelpers.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::parabolic::unittest
{
namespace
{
const std::string kTri3MeshType{"TRI3"};
const std::string kInternalThermalEnergyCriterionName{"Internal Thermal Energy"};
const std::string kTimeIntegratedStateAverageName{"integrated average temperature"};

Teuchos::ParameterList create_base_problem_parameters(const Plato::OrdinalType aNumSteps = 0)
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");
    tParameterList.set("PDE Constraint", "Parabolic");
    tParameterList.set("Physics", "Thermal");
    tParameterList.set("Output File", "test_solution_output.txt");

    tParameterList.sublist("Parabolic").sublist("Penalty Function").set("Exponent", 1.0);

    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Element Block", "body");
    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Material Model", "tapioca");

    tParameterList.sublist("Material Models")
        .sublist("tapioca")
        .sublist("Thermal Conduction")
        .set("Thermal Conductivity", 1.0);
    tParameterList.sublist("Material Models")
        .sublist("tapioca")
        .sublist("Thermal Mass")
        .set("Temperature Dependent", false);
    tParameterList.sublist("Material Models").sublist("tapioca").sublist("Thermal Mass").set("Specific Heat", 1.0);
    tParameterList.sublist("Material Models").sublist("tapioca").sublist("Thermal Mass").set("Mass Density", 1.0);

    tParameterList.sublist("Time Integration").set("Number Time Steps", aNumSteps);

    return tParameterList;
}

void append_internal_thermal_energy_criterion_to_parameter_list(Teuchos::ParameterList& aParamList)
{
    aParamList.sublist("Criteria").sublist(kInternalThermalEnergyCriterionName).set("Type", "Scalar Function");
    aParamList.sublist("Criteria")
        .sublist(kInternalThermalEnergyCriterionName)
        .set("Scalar Function Type", "Internal Thermal Energy");
    aParamList.sublist("Criteria")
        .sublist(kInternalThermalEnergyCriterionName)
        .sublist("Penalty Function")
        .set("Type", "SIMP");
    aParamList.sublist("Criteria")
        .sublist(kInternalThermalEnergyCriterionName)
        .sublist("Penalty Function")
        .set("Exponent", 1.0);
    aParamList.sublist("Criteria")
        .sublist(kInternalThermalEnergyCriterionName)
        .sublist("Penalty Function")
        .set("Minimum Value", 1e-16);
}

void append_time_integrated_state_average_criterion_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                                      const std::string& aNodeSet)
{
    aParamList.sublist("Criteria")
        .sublist(kTimeIntegratedStateAverageName)
        .set("Type", "Time Integrated State Average");
    aParamList.sublist("Criteria").sublist(kTimeIntegratedStateAverageName).set("Nodeset", aNodeSet);
    aParamList.sublist("Criteria").sublist(kTimeIntegratedStateAverageName).set("State Component", 0);
}

void append_fixed_temperature_boundary_conditions_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                                    const std::string& aSideSet,
                                                                    const Plato::Scalar aTemperatureValue = 0.0)
{
    const std::string tName = "Temperature Boundary Condition";
    const std::string tBCType = "Essential Boundary Conditions";
    aParamList.sublist(tBCType).sublist(tName).set("Type", "Fixed Value");
    aParamList.sublist(tBCType).sublist(tName).set("Index", 0);
    aParamList.sublist(tBCType).sublist(tName).set("Sides", "x-");
    aParamList.sublist(tBCType).sublist(tName).set("Value", aTemperatureValue);
}

void append_applied_flux_boundary_conditions_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                               const std::string& aSideSet,
                                                               Plato::Scalar aPrescribedFlux)
{
    const std::string tName = "Flux Boundary Condition";
    const std::string tBCType = "Natural Boundary Conditions";
    aParamList.sublist(tBCType).sublist(tName).set("Type", "Uniform");
    aParamList.sublist(tBCType).sublist(tName).set("Value", aPrescribedFlux);
    aParamList.sublist(tBCType).sublist(tName).set("Sides", aSideSet);
}

template <typename ElementType>
struct CreateParabolicThermalProblem
{
    auto operator()(const Plato::Mesh& aMesh, Teuchos::ParameterList& aParameterList) const
    {
        return parabolic::Problem<Plato::Thermal<ElementType>>(aMesh, aParameterList,
                                                               Plato::TestHelpers::dummy_comm_machine());
    }
};
}  // namespace

TEUCHOS_UNIT_TEST(ParabolicProblem, InternalThermalEnergyCriterionGradientPassesGradientCheck)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 2;
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    Teuchos::ParameterList tParamList = create_base_problem_parameters(tNumAnalysisSteps);
    append_internal_thermal_energy_criterion_to_parameter_list(tParamList);
    append_fixed_temperature_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x-");
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x+", tAppliedFlux);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    constexpr Plato::Scalar tControlValue{0.5};
    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 5, .mInitialStepSize = 0.01};
    constexpr Plato::Scalar tTruncationErrorTolerance{1.2e-1};
    Plato::TestHelpers::check_gradient_over_mesh(tParamList, CreateParabolicThermalProblem<Plato::Tri3>{},
                                                 kInternalThermalEnergyCriterionName, tMesh, tControlValue,
                                                 tGradientCheckParameters, tTruncationErrorTolerance, out, success);
}

TEUCHOS_UNIT_TEST(ParabolicProblem, TimeIntegratedStateAverageFunctionPassesGradientCheck_TempOnSurfaceWithFlux)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 4;
    Teuchos::ParameterList tParamList = create_base_problem_parameters(tNumAnalysisSteps);
    const std::string tNodeSet = "x+";
    append_time_integrated_state_average_criterion_to_parameter_list(tParamList, tNodeSet);
    append_fixed_temperature_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x-");
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, tNodeSet, tAppliedFlux);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    constexpr Plato::Scalar tControlValue{0.5};
    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 6, .mInitialStepSize = 0.1};
    constexpr Plato::Scalar tTruncationErrorTolerance{6e-2};
    Plato::TestHelpers::check_gradient_over_mesh(tParamList, CreateParabolicThermalProblem<Plato::Tri3>{},
                                                 kTimeIntegratedStateAverageName, tMesh, tControlValue,
                                                 tGradientCheckParameters, tTruncationErrorTolerance, out, success);
}

TEUCHOS_UNIT_TEST(ParabolicProblem, TimeIntegratedStateAverageFunctionPassesGradientCheck_TempOnSurfaceOppositeFlux)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 4;
    Teuchos::ParameterList tParamList = create_base_problem_parameters(tNumAnalysisSteps);
    append_time_integrated_state_average_criterion_to_parameter_list(tParamList, /*aNodeSet=*/"x-");
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x+", tAppliedFlux);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    constexpr Plato::Scalar tControlValue{0.5};
    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 5, .mInitialStepSize = 0.1};
    constexpr Plato::Scalar tTruncationErrorTolerance{5e-2};
    Plato::TestHelpers::check_gradient_over_mesh(tParamList, CreateParabolicThermalProblem<Plato::Tri3>{},
                                                 kTimeIntegratedStateAverageName, tMesh, tControlValue,
                                                 tGradientCheckParameters, tTruncationErrorTolerance, out, success);
}
}  // namespace plato::parabolic::unittest
