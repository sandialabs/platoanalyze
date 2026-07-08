#include <mpi.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include "element/Tri3.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "problem/Geometrical.hpp"
#include "problem/geometric/Problem.hpp"
#include "test_utilities/PlatoMPITestHelpers.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::problem::geometric::unittest
{
namespace
{
const std::string kVolumeCriterionName{"Volume"};

auto create_problem_param_list() -> Teuchos::ParameterList
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

void append_volume_criterion_to_parameter_list(Teuchos::ParameterList& aParamList)
{
    aParamList.sublist("Criteria").sublist(kVolumeCriterionName).set("Type", "Scalar Function");
    aParamList.sublist("Criteria").sublist(kVolumeCriterionName).set("Scalar Function Type", "Volume");
    aParamList.sublist("Criteria").sublist(kVolumeCriterionName).sublist("Penalty Function").set("Type", "SIMP");
    aParamList.sublist("Criteria").sublist(kVolumeCriterionName).sublist("Penalty Function").set("Exponent", 1.0);
    aParamList.sublist("Criteria")
        .sublist(kVolumeCriterionName)
        .sublist("Penalty Function")
        .set("Minimum Value", 1e-16);
}
}  // namespace

TEUCHOS_UNIT_TEST(GeometricProblem, VolumeCriterionMatchesExpected)
{
    constexpr Plato::OrdinalType tNumElementsPerDim = 1;
    constexpr Plato::Scalar tBoxDimension = 21.0;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tBoxDimension, tNumElementsPerDim, tBoxDimension,
                                                        tNumElementsPerDim, tBoxDimension, tNumElementsPerDim);
    const auto tNumNodes = tMesh->NumNodes();

    auto tParamList = create_problem_param_list();
    append_volume_criterion_to_parameter_list(tParamList);

    Problem<Plato::Geometrical<Plato::Tri3>> tProblem(tMesh, tParamList, Plato::TestHelpers::duplicate_comm_world());

    constexpr auto tGoldVolume = tBoxDimension * tBoxDimension;
    const Plato::ScalarVector tControl("control", tNumNodes);
    {
        Plato::blas1::fill(static_cast<Plato::Scalar>(1.0), tControl);

        const auto tValue = tProblem.criterionValue(tControl, Plato::Solutions{}, kVolumeCriterionName);
        TEST_EQUALITY(tValue, tGoldVolume);
    }

    {
        constexpr Plato::Scalar tControlValue = 0.86;
        Plato::blas1::fill(tControlValue, tControl);

        const auto tValue = tProblem.criterionValue(tControl, Plato::Solutions{}, kVolumeCriterionName);
        TEST_EQUALITY(tValue, tControlValue * tGoldVolume);
    }
}
}  // namespace plato::problem::geometric::unittest
