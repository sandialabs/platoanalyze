#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <exception>
#include <stdexcept>
#include <vector>

#include "core_types/PlatoTypes.hpp"
#include "domain/SpatialModel.hpp"
#include "element/ThermalElement.hpp"
#include "element/Tri3.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/Thermal.hpp"
#include "problem/parabolic/TimeIntegratedStateAverage.hpp"
#include "problem/parabolic/test_utilities/CommonInputParameters.hpp"
#include "problem/parabolic/test_utilities/DataUtilities.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::parabolic::unittest
{
namespace
{
const std::string kTri3MeshType{"TRI3"};
const std::string kCriterionName{"integrated average temperature"};

template <typename ElementType>
struct CreateTimeIntegratedStateAverageCriterion
{
    auto operator()(const plato::domain::SpatialModel& aSpatialModel,
                    Plato::DataMap& aDataMap,
                    Teuchos::ParameterList& aParameterList) const
    {
        return plato::parabolic::TimeIntegratedStateAverage<Plato::Thermal<typename ElementType::TopoElementType>>(
            aSpatialModel, aDataMap, aParameterList, kCriterionName);
    }
};

Plato::Solutions multi_step_solution_from_vector(const std::vector<std::vector<double>>& aStatesVector)
{
    const auto tStateMultiVector = test_utilities::multi_dimension_view_from_vector(aStatesVector);
    Plato::Solutions tSolution(std::string{}, std::string{});
    tSolution.set("State", tStateMultiVector);

    return tSolution;
}

void test_criterion_value_against_gold(const std::vector<std::vector<double>> aStatesVector,
                                       const Plato::Scalar aGoldValue,
                                       Teuchos::FancyOStream& aOutStream,
                                       bool& aSuccess)
{
    using ElementType = typename Plato::ThermalElement<Plato::Tri3>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();
    const auto tNumDofs = ElementType::mNumSpatialDims * tNumNodes;

    const std::string tNodeSet{"x-"};
    Teuchos::ParameterList tParamList = test_utilities::create_base_thermal_problem_parameters();
    test_utilities::append_time_integrated_state_average_criterion_to_parameter_list(tParamList, kCriterionName,
                                                                                     tNodeSet);

    const Plato::Solutions tSolution = multi_step_solution_from_vector(aStatesVector);

    const Plato::ScalarVector tControl("control of ones", tNumNodes);
    Plato::blas1::fill(1.0, tControl);

    constexpr Plato::Scalar tTimeStep = 1.0;
    const auto tValue = Plato::TestHelpers::compute_criterion_over_mesh<ElementType>(
        CreateTimeIntegratedStateAverageCriterion<ElementType>{}, tMesh, tParamList, tSolution, tControl, tTimeStep);

    TEUCHOS_TEST_EQUALITY(tValue, aGoldValue, aOutStream, aSuccess);
}
}  // namespace

TEUCHOS_UNIT_TEST(TimeIntegratedStateAverage, ConstructorParsingErrors)
{
    using ElementType = typename Plato::ThermalElement<Plato::Tri3>;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    Teuchos::ParameterList tBaseParamList = test_utilities::create_base_thermal_problem_parameters();
    tBaseParamList.sublist("Criteria").sublist(kCriterionName).set("Type", "Time Integrated State Average");

    Plato::DataMap tDataMap;
    const auto tParsedDomains = plato::domain::parse_domains(tBaseParamList, tMesh);
    plato::domain::SpatialModel tSpatialModel(tMesh, tParsedDomains, tDataMap);

    // no 'Nodeset' defined
    {
        auto tParamList = tBaseParamList;
        tParamList.sublist("Criteria").sublist(kCriterionName).set("State Component", 0);
        TEST_THROW([[maybe_unused]] const auto tCriterion =
                       CreateTimeIntegratedStateAverageCriterion<ElementType>{}(tSpatialModel, tDataMap, tParamList),
                   std::exception);
    }

    // specified 'Nodeset' not in mesh
    {
        auto tParamList = tBaseParamList;
        tParamList.sublist("Criteria").sublist(kCriterionName).set("State Component", 0);
        tParamList.sublist("Criteria").sublist(kCriterionName).set("Nodeset", "somewhere");
        TEST_THROW([[maybe_unused]] const auto tCriterion =
                       CreateTimeIntegratedStateAverageCriterion<ElementType>{}(tSpatialModel, tDataMap, tParamList),
                   std::runtime_error);
    }

    // no 'State Component' defined
    {
        auto tParamList = tBaseParamList;
        tParamList.sublist("Criteria").sublist(kCriterionName).set("Nodeset", "x-");
        TEST_THROW([[maybe_unused]] const auto tCriterion =
                       CreateTimeIntegratedStateAverageCriterion<ElementType>{}(tSpatialModel, tDataMap, tParamList),
                   std::exception);
    }

    // specified 'State Component' out of range of DOFs for physics
    {
        auto tParamList = tBaseParamList;
        tParamList.sublist("Criteria").sublist(kCriterionName).set("State Component", 1);
        tParamList.sublist("Criteria").sublist(kCriterionName).set("Nodeset", "x-");
        TEST_THROW([[maybe_unused]] const auto tCriterion =
                       CreateTimeIntegratedStateAverageCriterion<ElementType>{}(tSpatialModel, tDataMap, tParamList),
                   std::runtime_error);
    }

    // properly defined
    {
        auto tParamList = tBaseParamList;
        tParamList.sublist("Criteria").sublist(kCriterionName).set("State Component", 0);
        tParamList.sublist("Criteria").sublist(kCriterionName).set("Nodeset", "x-");
        TEST_NOTHROW([[maybe_unused]] const auto tCriterion =
                         CreateTimeIntegratedStateAverageCriterion<ElementType>{}(tSpatialModel, tDataMap, tParamList));
    }
}

TEUCHOS_UNIT_TEST(TimeIntegratedStateAverage, ZeroTemperatureGivesZeroValue)
{
    constexpr Plato::OrdinalType tNumDofs = 8;  // for the mesh used in test_criterion_value_against_gold
    const std::vector<std::vector<Plato::Scalar>> tStatesVector{
        std::vector<double>(tNumDofs, 0.0), std::vector<double>(tNumDofs, 0.0), std::vector<double>(tNumDofs, 0.0)};
    constexpr double tGoldValue = 0.0;  // first and last step values divided by 2 in trapezoid rule
    test_criterion_value_against_gold(tStatesVector, tGoldValue, out, success);
}

TEUCHOS_UNIT_TEST(TimeIntegratedStateAverage, PrescribedTemperatureGivesExpectedValue)
{
    const std::vector<std::vector<Plato::Scalar>> tStatesVector{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                {86.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    constexpr double tGoldValue =
        53.5 / 2.0;  // node set has 2 nodes (used in average), first step divided by 2 from trapezoid rule
    test_criterion_value_against_gold(tStatesVector, tGoldValue, out, success);
}

TEUCHOS_UNIT_TEST(TimeIntegratedStateAverage, PrescribedTemperatureGivesExpectedValue_MultipleSteps)
{
    const std::vector<std::vector<Plato::Scalar>> tStatesVector{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                {86.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                {77.0, 93.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                {88.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    constexpr double tGoldValue =
        170.0;  // node set has 2 nodes (used in average), first step divided by 2 from trapezoid rule
    test_criterion_value_against_gold(tStatesVector, tGoldValue, out, success);
}

}  // namespace plato::parabolic::unittest
