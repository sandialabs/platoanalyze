#include <mpi.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <plato/test_utilities/GradientChecker.hpp>
#include <string>

#include "element/Tri3.hpp"
#include "problem/Thermal.hpp"
#include "problem/parabolic/Problem.hpp"
#include "problem/parabolic/test_utilities/CommonInputParameters.hpp"
#include "test_utilities/PlatoGradientCheckTestHelpers.hpp"
#include "test_utilities/PlatoMPITestHelpers.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::parabolic::unittest
{
namespace
{
const std::string kTri3MeshType{"TRI3"};
const std::string kInternalThermalEnergyCriterionName{"Internal Thermal Energy"};
const std::string kTimeIntegratedStateAverageName{"integrated average temperature"};

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
                                                               Plato::TestHelpers::duplicate_comm_world());
    }
};
}  // namespace

TEUCHOS_UNIT_TEST(ParabolicProblem, InternalThermalEnergyCriterionGradientPassesGradientCheck)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 2;
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    Teuchos::ParameterList tParamList = test_utilities::create_base_thermal_problem_parameters();
    tParamList.sublist("Time Integration").set("Number Time Steps", tNumAnalysisSteps);
    append_internal_thermal_energy_criterion_to_parameter_list(tParamList);
    append_fixed_temperature_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x-");
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x+", tAppliedFlux);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 5, .mInitialStepSize = 0.01};
    constexpr Plato::Scalar tTruncationErrorTolerance{1.2e-1};

    constexpr Plato::Scalar tControlValue{0.5};
    const std::valarray<Plato::Scalar> tControl(tControlValue, tMesh->NumNodes());

    Plato::TestHelpers::check_control_gradient(
        Plato::TestHelpers::make_criterion_gradient_checker(
            CreateParabolicThermalProblem<Plato::Tri3>{}(tMesh, tParamList), kInternalThermalEnergyCriterionName),
        tGradientCheckParameters, tControl, tTruncationErrorTolerance, out, success);
}

TEUCHOS_UNIT_TEST(ParabolicProblem, TimeIntegratedStateAverageFunctionPassesGradientCheck_TempOnSurfaceWithFlux)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 4;
    Teuchos::ParameterList tParamList = test_utilities::create_base_thermal_problem_parameters();
    tParamList.sublist("Time Integration").set("Number Time Steps", tNumAnalysisSteps);
    const std::string tNodeSet = "x+";
    test_utilities::append_time_integrated_state_average_criterion_to_parameter_list(
        tParamList, kTimeIntegratedStateAverageName, tNodeSet);
    append_fixed_temperature_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x-");
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, tNodeSet, tAppliedFlux);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 6, .mInitialStepSize = 0.1};
    constexpr Plato::Scalar tTruncationErrorTolerance{6e-2};

    constexpr Plato::Scalar tControlValue{0.5};
    const std::valarray<Plato::Scalar> tControl(tControlValue, tMesh->NumNodes());

    Plato::TestHelpers::check_control_gradient(
        Plato::TestHelpers::make_criterion_gradient_checker(
            CreateParabolicThermalProblem<Plato::Tri3>{}(tMesh, tParamList), kTimeIntegratedStateAverageName),
        tGradientCheckParameters, tControl, tTruncationErrorTolerance, out, success);
}

TEUCHOS_UNIT_TEST(ParabolicProblem, TimeIntegratedStateAverageFunctionPassesGradientCheck_TempOnSurfaceOppositeFlux)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 4;
    Teuchos::ParameterList tParamList = test_utilities::create_base_thermal_problem_parameters();
    tParamList.sublist("Time Integration").set("Number Time Steps", tNumAnalysisSteps);
    test_utilities::append_time_integrated_state_average_criterion_to_parameter_list(
        tParamList, kTimeIntegratedStateAverageName, /*aNodeSet=*/"x-");
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x+", tAppliedFlux);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 5, .mInitialStepSize = 0.1};
    constexpr Plato::Scalar tTruncationErrorTolerance{5e-2};

    constexpr Plato::Scalar tControlValue{0.5};
    const std::valarray<Plato::Scalar> tControl(tControlValue, tMesh->NumNodes());

    Plato::TestHelpers::check_control_gradient(
        Plato::TestHelpers::make_criterion_gradient_checker(
            CreateParabolicThermalProblem<Plato::Tri3>{}(tMesh, tParamList), kTimeIntegratedStateAverageName),
        tGradientCheckParameters, tControl, tTruncationErrorTolerance, out, success);
}
}  // namespace plato::parabolic::unittest
