#include <mpi.h>

#include <Kokkos_StdAlgorithms.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <algorithm>
#include <cmath>
#include <plato/test_utilities/GradientChecker.hpp>
#include <stdexcept>
#include <string>
#include <valarray>

#include "boundary_conditions/EssentialBCs.hpp"
#include "domain/SpatialModel.hpp"
#include "element/Tri3.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/elliptic/VectorFunction.hpp"
#include "problem/elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"
#include "problem/elliptic/finite_deformation_mechanics/Problem.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"
#include "utilities/ParallelComm.hpp"

namespace plato::elliptic::finite_deformation_mechanics::unittest
{
namespace
{
Teuchos::ParameterList create_param_list(const Plato::OrdinalType aNumSteps = 0, const Plato::Scalar aTolerance = 0)
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");
    tParameterList.set("PDE Constraint", "Elliptic");
    tParameterList.set("Physics", "Finite Deformation Mechanics");
    tParameterList.set("Output File", "test_solution_output.txt");

    tParameterList.sublist("Elliptic").sublist("Penalty Function").set("Exponent", 1.0);

    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Element Block", "body");
    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Material Model", "Pudding");

    tParameterList.sublist("Material Models")
        .sublist("Pudding")
        .sublist("Neo Hookean Hyperelastic")
        .set("Bulk Modulus", 0.5);
    tParameterList.sublist("Material Models")
        .sublist("Pudding")
        .sublist("Neo Hookean Hyperelastic")
        .set("Shear Modulus", 0.375);

    tParameterList.sublist("Time Integration").set("Number Time Steps", aNumSteps);

    tParameterList.sublist("Newton Iteration").set("Maximum Iterations", 8);
    tParameterList.sublist("Newton Iteration").set("Increment Tolerance", 1e-16);
    tParameterList.sublist("Newton Iteration").set("Residual Tolerance", aTolerance);

    return tParameterList;
}

void append_strain_energy_criterion_to_parameter_list(Teuchos::ParameterList& aParamList)
{
    aParamList.sublist("Criteria").sublist("Strain Energy").set("Type", "Scalar Function");
    aParamList.sublist("Criteria").sublist("Strain Energy").set("Scalar Function Type", "Strain Energy");
    aParamList.sublist("Criteria").sublist("Strain Energy").sublist("Penalty Function").set("Type", "SIMP");
    aParamList.sublist("Criteria").sublist("Strain Energy").sublist("Penalty Function").set("Exponent", 1.0);
    aParamList.sublist("Criteria").sublist("Strain Energy").sublist("Penalty Function").set("Minimum Value", 1e-16);
}

void append_variance_of_strain_invariant_criterion_to_parameter_list(Teuchos::ParameterList& aParamList)
{
    aParamList.sublist("Criteria").sublist("Strain Variance").set("Type", "Variance Function");
    aParamList.sublist("Criteria").sublist("Strain Variance").set("Field Variable", "Strain Invariant");
    aParamList.sublist("Criteria").sublist("Strain Variance").sublist("Penalty Function").set("Type", "SIMP");
    aParamList.sublist("Criteria").sublist("Strain Variance").sublist("Penalty Function").set("Exponent", 1.0);
    aParamList.sublist("Criteria").sublist("Strain Variance").sublist("Penalty Function").set("Minimum Value", 1e-16);
}

void append_fixed_displacement_boundary_conditions_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                                     const Plato::OrdinalType aNumDofs)
{
    for (Plato::OrdinalType tDof = 0; tDof < aNumDofs; tDof++)
    {
        const std::string tName = "Left Fix DOF " + std::to_string(tDof);
        aParamList.sublist("Essential Boundary Conditions").sublist(tName).set("Type", "Zero Value");
        aParamList.sublist("Essential Boundary Conditions").sublist(tName).set("Index", tDof);
        aParamList.sublist("Essential Boundary Conditions").sublist(tName).set("Sides", "x-");
    }
}

void append_applied_displacement_boundary_conditions_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                                       Plato::Scalar aPrescribedDisplacement)
{
    aParamList.sublist("Essential Boundary Conditions").sublist("Right X Disp").set("Type", "Fixed Value");
    aParamList.sublist("Essential Boundary Conditions").sublist("Right X Disp").set("Sides", "x+");
    aParamList.sublist("Essential Boundary Conditions").sublist("Right X Disp").set("Index", 0);
    aParamList.sublist("Essential Boundary Conditions").sublist("Right X Disp").set("Value", aPrescribedDisplacement);
}

void append_applied_load_boundary_conditions_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                               Plato::Scalar aPrescribedLoad,
                                                               Plato::OrdinalType aNumSteps)
{
    aParamList.sublist("Natural Boundary Conditions").sublist("Right X Load").set("Type", "Uniform");
    aParamList.sublist("Natural Boundary Conditions").sublist("Right X Load").set("Sides", "x+");

    const std::string tLoadString = std::string{"t*"} + std::to_string(aPrescribedLoad / aNumSteps);
    aParamList.sublist("Natural Boundary Conditions")
        .sublist("Right X Load")
        .set<Teuchos::Array<std::string>>("Values", Teuchos::Array<std::string>{"0.0", tLoadString});
}

Plato::Comm::Machine dummy_comm_machine()
{
    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    return Plato::Comm::Machine(myComm);
}

template <template <typename> typename ContainerT, typename ScalarT>
Plato::ScalarVectorT<ScalarT> create_device_view(const ContainerT<ScalarT>& aVector)
{
    Kokkos::View<ScalarT*, Kokkos::HostSpace> tHostView("host view", aVector.size());
    std::copy(begin(aVector), end(aVector), Kokkos::Experimental::begin(tHostView));
    return Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace(), tHostView);
}

template <typename ElementType>
void check_gradient_over_mesh(Teuchos::ParameterList& aParamList,
                              const std::string aCriterionName,
                              const Plato::Mesh& aMesh,
                              Plato::Scalar aTruncationErrorTolerance,
                              Teuchos::FancyOStream& aOutStream,
                              bool& aSuccess)
{
    Problem<FiniteDeformationMechanics<ElementType>> tProblem(aMesh, aParamList, dummy_comm_machine());

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

    std::valarray<Plato::Scalar> tPerturbationDirection(1.0, tNumNodes);
    tPerturbationDirection[0] = 0.0;
    const auto tPerturbationNorm = std::sqrt(std::inner_product(
        begin(tPerturbationDirection), end(tPerturbationDirection), begin(tPerturbationDirection), 0.0));
    tPerturbationDirection /= tPerturbationNorm;

    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{/*mStepDelta=*/0.1, /*mNumSteps=*/5,
                                                                                  /*mInitialStepSize=*/0.1};

    const auto tMaxTruncationError =
        tGradientChecker.maxFirstOrderTruncationError(tControl, tPerturbationDirection, tGradientCheckParameters);
    TEUCHOS_TEST_ASSERT(tMaxTruncationError < aTruncationErrorTolerance, aOutStream, aSuccess);
}
}  // namespace

TEUCHOS_UNIT_TEST(FiniteDeformationProblem, SolutionReducesResidualBelowTolerance)
{
    // set parameters
    constexpr Plato::OrdinalType tNumAnalysisSteps = 1;
    constexpr Plato::Scalar tTolerance = 1e-6;
    Teuchos::ParameterList tParamList = create_param_list(tNumAnalysisSteps, tTolerance);
    append_fixed_displacement_boundary_conditions_to_parameter_list(tParamList, /*aNumDofs=*/2);
    append_applied_displacement_boundary_conditions_to_parameter_list(tParamList,
                                                                      /*aPrescribedDisplacement=*/1.0);

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();

    const Plato::ScalarVector tControl("control", tNumNodes);
    Plato::blas1::fill(static_cast<Plato::Scalar>(1.0), tControl);

    // solve PDE
    Problem<FiniteDeformationMechanics<Plato::Tri3>> tProblem(tMesh, tParamList, dummy_comm_machine());
    const auto tStateSolution = tProblem.solution(tControl);

    // evaluate residual at solution
    const auto tAllStates = tStateSolution.get("State");
    const auto tLastState = Kokkos::subview(tAllStates, tNumAnalysisSteps, Kokkos::ALL());

    Plato::DataMap tDataMap;
    const auto tParsedDomains = plato::domain::parse_domains(tParamList, tMesh);
    plato::domain::SpatialModel tSpatialModel(tMesh, tParsedDomains, tDataMap);

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    Plato::Elliptic::VectorFunction<FiniteDeformationMechanics<typename ElementType::TopoElementType>> tPDE(
        tSpatialModel, tDataMap, tParamList, tParamList.get<std::string>("PDE Constraint"));

    const auto tResidual = tPDE.value(tLastState, tControl);

    // zero essential BCs values for accurate residual norm
    Plato::EssentialBCs<ElementType> tEssentialBCs(tParamList.sublist("Essential Boundary Conditions", false), tMesh);
    Plato::OrdinalVector tBcDofs;
    Plato::ScalarVector tBcValues;
    tEssentialBCs.get(tBcDofs, tBcValues);
    Kokkos::parallel_for(
        "Dirichlet BC imposition", Kokkos::RangePolicy<int>(0, tBcDofs.size()),
        KOKKOS_LAMBDA(int tBcOrdinal) { tResidual(tBcDofs[tBcOrdinal]) = 0.0; });

    // test residual norm
    const auto tResidualNorm = Plato::blas1::norm(tResidual);
    std::cout << "\n Residual Norm at converged state : " << tResidualNorm << "\n";
    TEST_ASSERT(tResidualNorm < tTolerance);
}

TEUCHOS_UNIT_TEST(FiniteDeformationProblem, UniaxialExtensionSolutionProducesAnalyticStresses)
{
    // set parameters
    constexpr Plato::OrdinalType tNumAnalysisSteps = 2;
    constexpr Plato::Scalar tTolerance = 1e-14;

    Teuchos::ParameterList tParamList = create_param_list(tNumAnalysisSteps, tTolerance);
    // add plottable to store stresses
    tParamList.sublist("Elliptic").set<Teuchos::Array<std::string>>("Plottable", Teuchos::Array<std::string>{"stress"});

    // apply correct boundary conditions
    constexpr Plato::Scalar tAppliedExtension{1.0};
    append_fixed_displacement_boundary_conditions_to_parameter_list(tParamList, /*aNumDofs=*/1);
    append_applied_displacement_boundary_conditions_to_parameter_list(tParamList,
                                                                      /*aPrescribedDisplacement=*/tAppliedExtension);

    tParamList.sublist("Essential Boundary Conditions").sublist("Right Y Fix").set("Type", "Zero Value");
    tParamList.sublist("Essential Boundary Conditions").sublist("Right Y Fix").set("Index", 1);
    tParamList.sublist("Essential Boundary Conditions").sublist("Right Y Fix").set("Sides", "y+");

    tParamList.sublist("Essential Boundary Conditions").sublist("Bottom Y Fix").set("Type", "Zero Value");
    tParamList.sublist("Essential Boundary Conditions").sublist("Bottom Y Fix").set("Index", 1);
    tParamList.sublist("Essential Boundary Conditions").sublist("Bottom Y Fix").set("Sides", "y-");

    tParamList.sublist("Essential Boundary Conditions").sublist("Top Y Fix").set("Type", "Zero Value");
    tParamList.sublist("Essential Boundary Conditions").sublist("Top Y Fix").set("Index", 1);
    tParamList.sublist("Essential Boundary Conditions").sublist("Top Y Fix").set("Sides", "y+");

    tParamList.sublist("Essential Boundary Conditions").sublist("Back Z Fix").set("Type", "Zero Value");
    tParamList.sublist("Essential Boundary Conditions").sublist("Back Z Fix").set("Index", 2);
    tParamList.sublist("Essential Boundary Conditions").sublist("Back Z Fix").set("Sides", "z-");

    tParamList.sublist("Essential Boundary Conditions").sublist("Front Z Fix").set("Type", "Zero Value");
    tParamList.sublist("Essential Boundary Conditions").sublist("Front Z Fix").set("Index", 2);
    tParamList.sublist("Essential Boundary Conditions").sublist("Front Z Fix").set("Sides", "z+");

    constexpr Plato::OrdinalType tMeshWidth = 4;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();

    const Plato::ScalarVector tControl("control", tNumNodes);
    Plato::blas1::fill(static_cast<Plato::Scalar>(1.0), tControl);

    Problem<FiniteDeformationMechanics<Plato::Tet4>> tProblem(tMesh, tParamList, dummy_comm_machine());
    const auto tStateSolution = tProblem.solution(tControl);
    const auto tDataMaps = tProblem.getDataMap();
    const auto tNumStates = tDataMaps.stateDataMaps.size();
    const auto tDataMap = tDataMaps.getState(tNumStates - 1);  // get last state data map

    TEST_EQUALITY(tDataMap.scalarMultiVectors.size(), 1);
    const std::string tStressName{"stress"};
    const auto tStressData = tDataMap.scalarMultiVectors.at(tStressName);

    const auto tNumCells = static_cast<Plato::OrdinalType>(tStressData.extent(0));

    constexpr Plato::Scalar tKappa = 0.5;  // MPa
    constexpr Plato::Scalar tMu = 0.375;   // MPa
    constexpr Plato::Scalar tDeformationGradient11 = tAppliedExtension + 1.0;
    const auto tGoldCauchyStress11 = 0.5 * tKappa * (tDeformationGradient11 - 1.0 / tDeformationGradient11) +
                                     2.0 / 3.0 * tMu * (tDeformationGradient11 * tDeformationGradient11 - 1.0) *
                                         std::pow(tDeformationGradient11, -5.0 / 3.0);
    const auto tGoldCauchyStress22 = 0.5 * tKappa * (tDeformationGradient11 - 1.0 / tDeformationGradient11) -
                                     1.0 / 3.0 * tMu * (tDeformationGradient11 * tDeformationGradient11 - 1.0) *
                                         std::pow(tDeformationGradient11, -5.0 / 3.0);
    const auto tGoldCauchyStress33 = tGoldCauchyStress22;

    constexpr Plato::Scalar tTestTolerance{1.0e-14};

    const auto tHostStress = Plato::TestHelpers::get(tStressData);
    for (Plato::OrdinalType tCellOrdinal = 0; tCellOrdinal < tNumCells; tCellOrdinal++)
    {
        TEST_FLOATING_EQUALITY(tHostStress(tCellOrdinal, 0), tGoldCauchyStress11, tTestTolerance);
        TEST_FLOATING_EQUALITY(tHostStress(tCellOrdinal, 1), tGoldCauchyStress22, tTestTolerance);
        TEST_FLOATING_EQUALITY(tHostStress(tCellOrdinal, 2), tGoldCauchyStress33, tTestTolerance);
        TEST_ASSERT(tHostStress(tCellOrdinal, 3) < tTestTolerance);
        TEST_ASSERT(tHostStress(tCellOrdinal, 4) < tTestTolerance);
        TEST_ASSERT(tHostStress(tCellOrdinal, 5) < tTestTolerance);
    }
}

TEUCHOS_UNIT_TEST(FiniteDeformationProblem, ValueProducesExpectedUniaxialStrainEnergy)
{
    // set parameters
    constexpr Plato::OrdinalType tNumAnalysisSteps = 2;
    constexpr Plato::Scalar tTolerance = 1e-6;

    Teuchos::ParameterList tParamList = create_param_list(tNumAnalysisSteps, tTolerance);
    append_fixed_displacement_boundary_conditions_to_parameter_list(tParamList, /*aNumDofs=*/2);
    append_strain_energy_criterion_to_parameter_list(tParamList);
    append_applied_displacement_boundary_conditions_to_parameter_list(tParamList,
                                                                      /*aPrescribedDisplacement=*/1.0);
    // fix additional DOFs to make pure uniaxial displacement field
    tParamList.sublist("Essential Boundary Conditions").sublist("Right Y Fix").set("Type", "Zero Value");
    tParamList.sublist("Essential Boundary Conditions").sublist("Right Y Fix").set("Index", 1);
    tParamList.sublist("Essential Boundary Conditions").sublist("Right Y Fix").set("Sides", "x+");

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();

    constexpr Plato::Scalar tGoldValue{0.3479187954258798};  // energy is constant throughout domain with unit area
    // control of 1
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(static_cast<Plato::Scalar>(1.0), tControl);

        Problem<FiniteDeformationMechanics<Plato::Tri3>> tProblem(tMesh, tParamList, dummy_comm_machine());
        const auto tStateSolution = tProblem.solution(tControl);

        const auto tValue = tProblem.criterionValue(tControl, tStateSolution, "Strain Energy");
        TEST_FLOATING_EQUALITY(tValue, tGoldValue, 1e-14);
    }

    // control of 0.5
    {
        constexpr Plato::Scalar tControlVal{0.5};
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(static_cast<Plato::Scalar>(tControlVal), tControl);

        Problem<FiniteDeformationMechanics<Plato::Tri3>> tProblem(tMesh, tParamList, dummy_comm_machine());
        const auto tStateSolution = tProblem.solution(tControl);

        const auto tValue = tProblem.criterionValue(tControl, tStateSolution, "Strain Energy");
        TEST_FLOATING_EQUALITY(tValue, tControlVal * tGoldValue, 1e-14);
    }
}

TEUCHOS_UNIT_TEST(FiniteDeformationProblem, StrainEnergyCriterionGradientPassesGradientCheckSelfAdjoint)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 2;
    constexpr Plato::Scalar tTolerance = 1e-14;

    Teuchos::ParameterList tParamList = create_param_list(tNumAnalysisSteps, tTolerance);
    append_fixed_displacement_boundary_conditions_to_parameter_list(tParamList, /*aNumDofs=*/2);
    append_strain_energy_criterion_to_parameter_list(tParamList);
    append_applied_displacement_boundary_conditions_to_parameter_list(tParamList,
                                                                      /*aPrescribedDisplacement=*/1.0);
    tParamList.set("Self-Adjoint", true);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    constexpr Plato::Scalar tTruncationErrorTolerance{5e-2};
    check_gradient_over_mesh<Plato::Tri3>(tParamList, "Strain Energy", tMesh, tTruncationErrorTolerance, out, success);
}

TEUCHOS_UNIT_TEST(FiniteDeformationProblem, StrainEnergyCriterionGradientPassesGradientCheckNonSelfAdjoint)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 2;
    constexpr Plato::Scalar tTolerance = 1e-14;

    Teuchos::ParameterList tParamList = create_param_list(tNumAnalysisSteps, tTolerance);
    append_fixed_displacement_boundary_conditions_to_parameter_list(tParamList, /*aNumDofs=*/2);
    append_strain_energy_criterion_to_parameter_list(tParamList);
    append_applied_load_boundary_conditions_to_parameter_list(tParamList,
                                                              /*aPrescribedLoad=*/1.0e-2,
                                                              /*aNumSteps=*/tNumAnalysisSteps);
    tParamList.set("Self-Adjoint", false);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    constexpr Plato::Scalar tTruncationErrorTolerance{5e-2};
    check_gradient_over_mesh<Plato::Tri3>(tParamList, "Strain Energy", tMesh, tTruncationErrorTolerance, out, success);
}

TEUCHOS_UNIT_TEST(FiniteDeformationProblem, VarianceCriterionGradientPassesGradientCheckNonSelfAdjoint)
{
    constexpr Plato::OrdinalType tNumAnalysisSteps = 2;
    constexpr Plato::Scalar tTolerance = 1e-14;

    Teuchos::ParameterList tParamList = create_param_list(tNumAnalysisSteps, tTolerance);
    append_fixed_displacement_boundary_conditions_to_parameter_list(tParamList, /*aNumDofs=*/2);
    append_variance_of_strain_invariant_criterion_to_parameter_list(tParamList);
    append_applied_displacement_boundary_conditions_to_parameter_list(tParamList,
                                                                      /*aPrescribedDisplacement=*/1.0);
    tParamList.set("Self-Adjoint", false);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    constexpr Plato::Scalar tTruncationErrorTolerance{5e-3};
    check_gradient_over_mesh<Plato::Tri3>(tParamList, "Strain Variance", tMesh, tTruncationErrorTolerance, out,
                                          success);
}
}  // namespace plato::elliptic::finite_deformation_mechanics::unittest
