#include <mpi.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <plato/test_utilities/GradientChecker.hpp>
#include <valarray>

#include "element/Tri3.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/Thermal.hpp"
#include "problem/parabolic/Problem.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"
#include "utilities/ParallelComm.hpp"

namespace plato::parabolic::unittest
{
namespace
{
const std::string kTri3MeshType{"TRI3"};
const std::string kInternalThermalEnergyCriterionName{"Internal Thermal Energy"};

Plato::Comm::Machine dummy_comm_machine()
{
    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    return Plato::Comm::Machine(myComm);
}

Teuchos::ParameterList create_base_problem_parameters(const Plato::OrdinalType aNumSteps = 0)
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");
    tParameterList.set("PDE Constraint", "Parabolic");
    tParameterList.set("Physics", "Thermal");

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

void append_fixed_temperature_boundary_conditions_to_parameter_list(Teuchos::ParameterList& aParamList)
{
    const std::string tName = "Temperature Boundary Condition";
    const std::string tBCType = "Essential Boundary Conditions";
    aParamList.sublist(tBCType).sublist(tName).set("Type", "Fixed Value");
    aParamList.sublist(tBCType).sublist(tName).set("Index", 0);
    aParamList.sublist(tBCType).sublist(tName).set("Sides", "x-");
    aParamList.sublist(tBCType).sublist(tName).set("Value", 0.0);
}

void append_applied_flux_boundary_conditions_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                               Plato::Scalar aPrescribedFlux)
{
    const std::string tName = "Flux Boundary Condition";
    const std::string tBCType = "Natural Boundary Conditions";
    aParamList.sublist(tBCType).sublist(tName).set("Type", "Uniform");
    aParamList.sublist(tBCType).sublist(tName).set("Value", aPrescribedFlux);
    aParamList.sublist(tBCType).sublist(tName).set("Sides", "x+");
}

// TODO: Pull out since this is used in finite deformation mechanics as well
template <template <typename> typename ContainerT, typename ScalarT>
Plato::ScalarVectorT<ScalarT> create_device_view(const ContainerT<ScalarT>& aVector)
{
    Kokkos::View<ScalarT*, Kokkos::HostSpace> tHostView("host view", aVector.size());
    std::copy(begin(aVector), end(aVector), Kokkos::Experimental::begin(tHostView));
    return Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostView);
}

// TODO: Pull out since this is used in finite deformation mechanics as well
template <typename ElementType>
void check_gradient_over_mesh(Teuchos::ParameterList& aParamList,
                              const std::string aCriterionName,
                              const Plato::Mesh& aMesh,
                              Plato::Scalar aTruncationErrorTolerance,
                              Teuchos::FancyOStream& aOutStream,
                              bool& aSuccess)
{
    parabolic::Problem<Plato::Thermal<ElementType>> tProblem(aMesh, aParamList, dummy_comm_machine());

    auto tCriterionValue = [&aCriterionName, &tProblem](const std::valarray<Plato::Scalar>& aControlVector)
    {
        const auto tControl = create_device_view(aControlVector);
        const auto tStateSolution = tProblem.solution(tControl);
        return tProblem.criterionValue(tControl, tStateSolution, aCriterionName);
    };

    auto tCriterionGradient = [&aCriterionName, &tProblem](const std::valarray<Plato::Scalar>& aControlVector,
                                                           const std::valarray<Plato::Scalar>& aDirection)
    {
        const auto tControl = create_device_view(aControlVector);
        const auto tStateSolution = tProblem.solution(tControl);
        const auto tGradient = tProblem.criterionGradient(tControl, tStateSolution, aCriterionName);
        const auto tHostGradient = Plato::TestHelpers::get(tGradient);
        return std::inner_product(Kokkos::Experimental::begin(tHostGradient), Kokkos::Experimental::end(tHostGradient),
                                  begin(aDirection), 0.0);
    };

    const plato::test_utilities::GradientChecker<std::valarray<Plato::Scalar>> tGradientChecker{tCriterionValue,
                                                                                                tCriterionGradient};

    const auto tNumNodes = aMesh->NumNodes();
    constexpr Plato::Scalar tControlVal{0.5};
    const std::valarray<Plato::Scalar> tControl(tControlVal, tNumNodes);

    // uniform perturbation
    std::valarray<Plato::Scalar> tPerturbationDirection(1.0, tNumNodes);
    const auto tPerturbationNorm = std::sqrt(std::inner_product(
        begin(tPerturbationDirection), end(tPerturbationDirection), begin(tPerturbationDirection), 0.0));
    tPerturbationDirection /= tPerturbationNorm;

    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 6, .mInitialStepSize = 0.1};

    const auto tMaxTruncationError =
        tGradientChecker.maxFirstOrderTruncationError(tControl, tPerturbationDirection, tGradientCheckParameters);
    TEUCHOS_TEST_ASSERT(tMaxTruncationError < aTruncationErrorTolerance, aOutStream, aSuccess);
    if (!aSuccess)
    {
        std::cout << tGradientChecker.table(tControl, tPerturbationDirection, tGradientCheckParameters);
    }
}
}  // namespace

TEUCHOS_UNIT_TEST(ParabolicProblem, InternalThermalEnergyCriterionGradientPassesGradientCheck)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 2;
    constexpr Plato::Scalar tAppliedFlux = 1.0;
    Teuchos::ParameterList tParamList = create_base_problem_parameters(tNumAnalysisSteps);
    append_internal_thermal_energy_criterion_to_parameter_list(tParamList);
    append_fixed_temperature_boundary_conditions_to_parameter_list(tParamList);
    append_applied_flux_boundary_conditions_to_parameter_list(tParamList, tAppliedFlux);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    constexpr Plato::Scalar tTruncationErrorTolerance{5e-2};
    check_gradient_over_mesh<Plato::Tri3>(tParamList, kInternalThermalEnergyCriterionName, tMesh,
                                          tTruncationErrorTolerance, out, success);
}
}  // namespace plato::parabolic::unittest
