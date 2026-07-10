#include <mpi.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <plato/utilities/GradientChecker.hpp>
#include <string>

#include "element/Hex8.hpp"
#include "element/Quad4.hpp"
#include "element/Tet4.hpp"
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
const std::string kTet4MeshType{"TET4"};
const std::string kQuad4MeshType{"QUAD4"};
const std::string kHex8MeshType{"HEX8"};

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

template <typename TopoElementType>
void run_internal_thermal_energy_gradient_check(const std::string& aElementType,
                                                const Plato::OrdinalType aMeshWidth,
                                                const Plato::OrdinalType aNumAnalysisSteps,
                                                const std::string& aFluxNodeSet,
                                                const Plato::Scalar aTruncationErrorTolerance,
                                                Teuchos::FancyOStream& aOutStream,
                                                bool& aSuccess)
{
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    Teuchos::ParameterList tParamList = test_utilities::create_base_thermal_problem_parameters();
    tParamList.sublist("Time Integration").set("Number Time Steps", aNumAnalysisSteps);
    append_internal_thermal_energy_criterion_to_parameter_list(tParamList);
    append_fixed_temperature_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x-");
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, /*aSideSet=*/"x+", tAppliedFlux);

    const auto tMesh = Plato::TestHelpers::get_box_mesh(aElementType, aMeshWidth);

    const plato::utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 5, .mInitialStepSize = 0.01};

    constexpr Plato::Scalar tControlValue{0.5};
    const std::valarray<Plato::Scalar> tControl(tControlValue, tMesh->NumNodes());

    Plato::TestHelpers::check_control_gradient(
        Plato::TestHelpers::make_criterion_gradient_checker(
            CreateParabolicThermalProblem<TopoElementType>{}(tMesh, tParamList), kInternalThermalEnergyCriterionName),
        tGradientCheckParameters, tControl, aTruncationErrorTolerance, aOutStream, aSuccess);
}

struct TimeIntegratedStateAverageNodeSets
{
    std::string mFluxNodeSet;
    std::string mFixedTemperatureNodeSet;
    std::string mMeasuredNodeSet;
};

template <typename TopoElementType>
void run_time_integrated_state_average_gradient_check(const std::string& aElementType,
                                                      const Plato::OrdinalType aMeshWidth,
                                                      const Plato::OrdinalType aNumAnalysisSteps,
                                                      const TimeIntegratedStateAverageNodeSets& aNodesets,
                                                      const Plato::Scalar aTruncationErrorTolerance,
                                                      Teuchos::FancyOStream& aOutStream,
                                                      bool& aSuccess)
{
    Teuchos::ParameterList tParamList = test_utilities::create_base_thermal_problem_parameters();
    tParamList.sublist("Time Integration").set("Number Time Steps", aNumAnalysisSteps);
    test_utilities::append_time_integrated_state_average_criterion_to_parameter_list(
        tParamList, kTimeIntegratedStateAverageName, aNodesets.mMeasuredNodeSet);
    if (!aNodesets.mFixedTemperatureNodeSet.empty())
    {
        append_fixed_temperature_boundary_conditions_to_parameter_list(tParamList,
                                                                       /*aSideSet=*/aNodesets.mFixedTemperatureNodeSet);
    }
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, aNodesets.mFluxNodeSet, tAppliedFlux);

    const auto tMesh = Plato::TestHelpers::get_box_mesh(aElementType, aMeshWidth);

    const plato::utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 5, .mInitialStepSize = 0.01};

    constexpr Plato::Scalar tControlValue{0.5};
    const std::valarray<Plato::Scalar> tControl(tControlValue, tMesh->NumNodes());

    Plato::TestHelpers::check_control_gradient(
        Plato::TestHelpers::make_criterion_gradient_checker(
            CreateParabolicThermalProblem<TopoElementType>{}(tMesh, tParamList), kTimeIntegratedStateAverageName),
        tGradientCheckParameters, tControl, aTruncationErrorTolerance, aOutStream, aSuccess);
}

}  // namespace

TEUCHOS_UNIT_TEST(ParabolicProblem, InternalThermalEnergyCriterionGradientChecks)
{
    // tri3
    {
        constexpr Plato::OrdinalType tNumAnalysisSteps = 6;
        constexpr Plato::OrdinalType tMeshWidth = 5;
        const std::string tFluxNodeSet{"x+"};
        constexpr Plato::Scalar tTruncationErrorTolerance{1.2e-1};
        run_internal_thermal_energy_gradient_check<Plato::Tri3>(kTri3MeshType, tMeshWidth, tNumAnalysisSteps,
                                                                tFluxNodeSet, tTruncationErrorTolerance, out, success);
    }

    // tet4
    {
        constexpr Plato::OrdinalType tNumAnalysisSteps = 3;
        constexpr Plato::OrdinalType tMeshWidth = 3;
        const std::string tFluxNodeSet{"z+"};
        constexpr Plato::Scalar tTruncationErrorTolerance{2.0e-2};
        run_internal_thermal_energy_gradient_check<Plato::Tet4>(kTet4MeshType, tMeshWidth, tNumAnalysisSteps,
                                                                tFluxNodeSet, tTruncationErrorTolerance, out, success);
    }

    // hex8
    {
        constexpr Plato::OrdinalType tNumAnalysisSteps = 3;
        constexpr Plato::OrdinalType tMeshWidth = 4;
        const std::string tFluxNodeSet{"y-"};
        constexpr Plato::Scalar tTruncationErrorTolerance{2.0e-2};
        run_internal_thermal_energy_gradient_check<Plato::Hex8>(kHex8MeshType, tMeshWidth, tNumAnalysisSteps,
                                                                tFluxNodeSet, tTruncationErrorTolerance, out, success);
    }
}

TEUCHOS_UNIT_TEST(ParabolicProblem, TimeIntegratedStateAverageGradientChecks)
{
    // tri3, measured on flux node set
    {
        constexpr Plato::OrdinalType tNumAnalysisSteps = 6;
        constexpr Plato::OrdinalType tMeshWidth = 4;
        const auto tNodeSets = TimeIntegratedStateAverageNodeSets{
            .mFluxNodeSet = "x+", .mFixedTemperatureNodeSet = "x-", .mMeasuredNodeSet = "x+"};
        constexpr Plato::Scalar tTruncationErrorTolerance{1.0e-2};
        run_time_integrated_state_average_gradient_check<Plato::Tri3>(
            kTri3MeshType, tMeshWidth, tNumAnalysisSteps, tNodeSets, tTruncationErrorTolerance, out, success);
    }

    // quad4, measured opposite flux node set, no fixed temperature BC
    {
        constexpr Plato::OrdinalType tNumAnalysisSteps = 6;
        constexpr Plato::OrdinalType tMeshWidth = 4;
        const auto tNodeSets = TimeIntegratedStateAverageNodeSets{
            .mFluxNodeSet = "y+", .mFixedTemperatureNodeSet = "", .mMeasuredNodeSet = "y-"};
        constexpr Plato::Scalar tTruncationErrorTolerance{2.0e-1};
        run_time_integrated_state_average_gradient_check<Plato::Quad4>(
            kQuad4MeshType, tMeshWidth, tNumAnalysisSteps, tNodeSets, tTruncationErrorTolerance, out, success);
    }

    // hex8, measured opposite flux node set, with fixed temperature BC
    {
        constexpr Plato::OrdinalType tNumAnalysisSteps = 3;
        constexpr Plato::OrdinalType tMeshWidth = 3;
        const auto tNodeSets = TimeIntegratedStateAverageNodeSets{
            .mFluxNodeSet = "z+", .mFixedTemperatureNodeSet = "y+", .mMeasuredNodeSet = "z-"};
        constexpr Plato::Scalar tTruncationErrorTolerance{5.0e-2};
        run_time_integrated_state_average_gradient_check<Plato::Hex8>(
            kHex8MeshType, tMeshWidth, tNumAnalysisSteps, tNodeSets, tTruncationErrorTolerance, out, success);
    }
}
}  // namespace plato::parabolic::unittest
