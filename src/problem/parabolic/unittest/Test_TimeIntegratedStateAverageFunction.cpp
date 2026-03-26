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
#include "problem/parabolic/TimeIntegratedStateAverageFunction.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::parabolic::unittest
{
namespace
{
const std::string kTri3MeshType{"TRI3"};
const std::string kCriterionName{"integrated average temperature"};

Teuchos::ParameterList create_base_param_list()
{
    const std::string tMaterialName = "tapioca";

    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");
    tParameterList.set("PDE Constraint", "Parabolic");
    tParameterList.set("Physics", "Thermal");

    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Element Block", "body");
    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Material Model", tMaterialName);

    return tParameterList;
}

void append_valid_criterion_param_list(Teuchos::ParameterList& aParamList, const std::string& aNodeSet)
{
    aParamList.sublist("Criteria").sublist(kCriterionName).set("Type", "Time Integrated State Average");
    aParamList.sublist("Criteria").sublist(kCriterionName).set("Nodeset", aNodeSet);
    aParamList.sublist("Criteria").sublist(kCriterionName).set("State Component", 0);
}

template <typename ElementType>
struct CreateTimeIntegratedStateAverageCriterion
{
    auto operator()(const plato::domain::SpatialModel& aSpatialModel,
                    Plato::DataMap& aDataMap,
                    Teuchos::ParameterList& aParameterList) const
    {
        return plato::parabolic::TimeIntegratedStateAverageFunction<
            Plato::Thermal<typename ElementType::TopoElementType>>(aSpatialModel, aDataMap, aParameterList,
                                                                   kCriterionName);
    }
};

Plato::Solutions multi_step_solution_from_vector(const std::vector<std::vector<double>>& aStatesVector)
{
    const Plato::OrdinalType tNumSteps = aStatesVector.size();
    assert(tNumSteps > 0);
    const Plato::OrdinalType tNumDofs = aStatesVector[0].size();
    Plato::ScalarMultiVector tStateMultiVector("state", tNumSteps, tNumDofs);
    for (Plato::OrdinalType tStep = 0; tStep < tNumSteps; tStep++)
    {
        const auto tStateView = Plato::TestHelpers::create_device_view(aStatesVector[tStep]);
        Kokkos::parallel_for(
            "multidimensional view", Kokkos::RangePolicy<int>(0, tNumDofs),
            KOKKOS_LAMBDA(Plato::OrdinalType tDofOrdinal) {
                tStateMultiVector(tStep, tDofOrdinal) = tStateView(tDofOrdinal);
            });
    }
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
    Teuchos::ParameterList tParamList = create_base_param_list();
    append_valid_criterion_param_list(tParamList, tNodeSet);

    const Plato::Solutions tSolution = multi_step_solution_from_vector(aStatesVector);

    const Plato::ScalarVector tControl("control of ones", tNumNodes);
    Plato::blas1::fill(1.0, tControl);

    constexpr Plato::Scalar tTimeStep = 1.0;
    const auto tValue = Plato::TestHelpers::compute_criterion_over_mesh<ElementType>(
        CreateTimeIntegratedStateAverageCriterion<ElementType>{}, tMesh, tParamList, tSolution, tControl, tTimeStep);

    TEUCHOS_TEST_EQUALITY(tValue, aGoldValue, aOutStream, aSuccess);
}
}  // namespace

TEUCHOS_UNIT_TEST(TimeIntegratedStateAverageFunction, ConstructorParsingErrors)
{
    using ElementType = typename Plato::ThermalElement<Plato::Tri3>;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTri3MeshType, tMeshWidth);

    Teuchos::ParameterList tBaseParamList = create_base_param_list();
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

TEUCHOS_UNIT_TEST(TimeIntegratedStateAverageFunction, ZeroTemperatureGivesZeroValue)
{
    constexpr Plato::OrdinalType tNumDofs = 8;  // for the mesh used in test_criterion_value_against_gold
    const std::vector<std::vector<Plato::Scalar>> tStatesVector{
        std::vector<double>(tNumDofs, 0.0), std::vector<double>(tNumDofs, 0.0), std::vector<double>(tNumDofs, 0.0)};
    constexpr double tGoldValue = 0.0;  // first and last step values divided by 2 in trapezoid rule
    test_criterion_value_against_gold(tStatesVector, tGoldValue, out, success);
}

TEUCHOS_UNIT_TEST(TimeIntegratedStateAverageFunction, PrescribedTemperatureGivesExpectedValue)
{
    const std::vector<std::vector<Plato::Scalar>> tStatesVector{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                {86.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    constexpr double tGoldValue = 53.5 / 2.0;  // first step divided by 2 from trapezoid rule
    test_criterion_value_against_gold(tStatesVector, tGoldValue, out, success);
}

TEUCHOS_UNIT_TEST(TimeIntegratedStateAverageFunction, PrescribedTemperatureGivesExpectedValue_MultipleSteps)
{
    const std::vector<std::vector<Plato::Scalar>> tStatesVector{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                {86.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                {77.0, 93.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                                                {88.0, 38.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}};
    constexpr double tGoldValue = 143.25;  // first and last step values divided by 2 in trapezoid rule
    test_criterion_value_against_gold(tStatesVector, tGoldValue, out, success);
}

}  // namespace plato::parabolic::unittest
