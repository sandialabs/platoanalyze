#include <mpi.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <plato/test_utilities/GradientChecker.hpp>
#include <string_view>
#include <valarray>

#include "element/Tet4.hpp"
#include "element/Tri3.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/Geometrical.hpp"
#include "problem/geometric/Problem.hpp"
#include "problem/geometric/test_utilities/MassPropertiesCriterionUtilities.hpp"
#include "test_utilities/PlatoGradientCheckTestHelpers.hpp"
#include "test_utilities/PlatoMPITestHelpers.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::problem::geometric::unittest
{
namespace
{
constexpr auto kVolumeCriterionName = std::string_view{"Volume"};
constexpr auto kMassPropertiesCriterionName = std::string_view{"Mass Properties"};

[[nodiscard]] auto create_problem_param_list() -> Teuchos::ParameterList
{
    Teuchos::ParameterList tParameterList;

    tParameterList.setName("Plato Problem");
    tParameterList.set("PDE Constraint", "Geometric");
    tParameterList.set("Physics", "Geometric");

    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Element Block", "body");
    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Material Model", "Catsup");

    tParameterList.sublist("Material Models").sublist("Catsup").set("Density", 1.27);

    return tParameterList;
}

void append_volume_criterion(Teuchos::ParameterList& aParamList)
{
    aParamList.sublist("Criteria").sublist(std::string{kVolumeCriterionName}).set("Type", "Scalar Function");
    aParamList.sublist("Criteria").sublist(std::string{kVolumeCriterionName}).set("Scalar Function Type", "Volume");
    aParamList.sublist("Criteria")
        .sublist(std::string{kVolumeCriterionName})
        .sublist("Penalty Function")
        .set("Type", "SIMP");
    aParamList.sublist("Criteria")
        .sublist(std::string{kVolumeCriterionName})
        .sublist("Penalty Function")
        .set("Exponent", 1.0);
    aParamList.sublist("Criteria")
        .sublist(std::string{kVolumeCriterionName})
        .sublist("Penalty Function")
        .set("Minimum Value", 0.0);
}

template <typename ElementType>
struct CreateGeometricProblem
{
    auto operator()(const Plato::Mesh& aMesh, Teuchos::ParameterList& aParameterList) const
    {
        return Problem<Plato::Geometrical<ElementType>>(aMesh, aParameterList,
                                                        Plato::TestHelpers::duplicate_comm_world());
    }
};
}  // namespace

TEUCHOS_UNIT_TEST(GeometricProblem, VolumeCriterionMatchesExpected)
{
    constexpr Plato::OrdinalType tNumElementsPerDim = 1;
    constexpr Plato::Scalar tBoxDimension = 21.0;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tBoxDimension, tNumElementsPerDim, tBoxDimension,
                                                        tNumElementsPerDim, tBoxDimension, tNumElementsPerDim);
    const auto tNumNodes = tMesh->NumNodes();

    auto tParamList = create_problem_param_list();
    append_volume_criterion(tParamList);

    Problem<Plato::Geometrical<Plato::Tri3>> tProblem(tMesh, tParamList, Plato::TestHelpers::duplicate_comm_world());

    constexpr auto tGoldVolume = tBoxDimension * tBoxDimension;
    const Plato::ScalarVector tControl("control", tNumNodes);
    {
        Plato::blas1::fill(static_cast<Plato::Scalar>(1.0), tControl);

        const auto tValue = tProblem.criterionValue(tControl, Plato::Solutions{}, std::string{kVolumeCriterionName});
        TEST_EQUALITY(tValue, tGoldVolume);
    }

    {
        constexpr Plato::Scalar tControlValue = 0.86;
        Plato::blas1::fill(tControlValue, tControl);

        const auto tValue = tProblem.criterionValue(tControl, Plato::Solutions{}, std::string{kVolumeCriterionName});
        TEST_EQUALITY(tValue, tControlValue * tGoldVolume);
    }
}

TEUCHOS_UNIT_TEST(GeometricProblem, VolumeCriterionPassesGradientCheck)
{
    auto tParamList = create_problem_param_list();
    append_volume_criterion(tParamList);
    tParamList.sublist("Criteria")
        .sublist(std::string{kVolumeCriterionName})
        .sublist("Penalty Function")
        .set("Exponent", 3.0);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 5, .mInitialStepSize = 0.1};
    constexpr Plato::Scalar tTruncationErrorTolerance{1e-2};

    constexpr Plato::Scalar tControlValue{0.6};
    const std::valarray<Plato::Scalar> tControl(tControlValue, tMesh->NumNodes());

    Plato::TestHelpers::check_control_gradient(
        Plato::TestHelpers::make_criterion_gradient_checker(CreateGeometricProblem<Plato::Tet4>{}(tMesh, tParamList),
                                                            std::string{kVolumeCriterionName}),
        tGradientCheckParameters, tControl, tTruncationErrorTolerance, out, success);
}

TEUCHOS_UNIT_TEST(GeometricProblem, MassPropertiesCriterionPassesGradientCheck)
{
    auto tParamList = create_problem_param_list();

    constexpr Plato::OrdinalType tExponent(2);

    constexpr Plato::OrdinalType tMeshWidth = 5;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    const plato::test_utilities::GradientCheckParameters tGradientCheckParameters{
        .mStepDelta = 0.1, .mNumSteps = 5, .mInitialStepSize = 0.1};
    constexpr Plato::Scalar tTruncationErrorTolerance{1e-2};

    constexpr Plato::Scalar tControlValue{0.86};
    const std::valarray<Plato::Scalar> tControl(tControlValue, tMesh->NumNodes());

    // non-normalized
    {
        tParamList.sublist("Criteria") = test_utilities::mass_properties_criterion(
            /*PropertyList*/ {"Mass", "CGx", "CGy", "CGz", "Ixx", "Iyy", "Izz", "Ixy", "Iyz"},
            /*WeightList*/ {2.0, 0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
            /*GoldList*/ {0.2, 0.05, 0.55, 0.75, 0.5, 0.5, 0.5, 0.3, 0.3}, tExponent);
        Plato::TestHelpers::check_control_gradient(
            Plato::TestHelpers::make_criterion_gradient_checker(
                CreateGeometricProblem<Plato::Tet4>{}(tMesh, tParamList), std::string{kMassPropertiesCriterionName}),
            tGradientCheckParameters, tControl, tTruncationErrorTolerance, out, success);
    }

    // normalized
    {
        tParamList.sublist("Criteria") = test_utilities::mass_properties_criterion(
            /*PropertyList*/ {"Mass", "CGx", "CGy", "CGz", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"},
            /*WeightList*/ {2.0, 0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
            /*GoldList*/ {0.2, 0.05, 0.55, 0.75, 5.4, 5.5, 5.4, -0.1, -0.1, -0.15}, tExponent);
        Plato::TestHelpers::check_control_gradient(
            Plato::TestHelpers::make_criterion_gradient_checker(
                CreateGeometricProblem<Plato::Tet4>{}(tMesh, tParamList), std::string{kMassPropertiesCriterionName}),
            tGradientCheckParameters, tControl, tTruncationErrorTolerance, out, success);
    }
}
}  // namespace plato::problem::geometric::unittest
