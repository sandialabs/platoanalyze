
#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "element/MechanicsElement.hpp"
#include "element/Tet4.hpp"
#include "element/Tri3.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "problem/Analyze_Diagnostics.hpp"
#include "problem/Geometrical.hpp"
#include "problem/geometric/GeometryScalarFunction.hpp"
#include "problem/geometric/MassMoment.hpp"
#include "problem/geometric/MassPropertiesFunction.hpp"
#include "problem/geometric/WeightedSumFunction.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace problem::geometric::unittest
{

namespace
{
constexpr Plato::Scalar kTolerance = 1e-15;
constexpr auto kMaterialName = std::string_view{"material"};
constexpr auto kSpatialModelName = std::string_view{"Spatial Model"};
constexpr auto kMaterialModelsName = std::string_view{"Material Models"};
constexpr auto kMassPropertiesName = std::string_view{"Mass Properties"};
[[nodiscard]] auto material_model() -> Teuchos::ParameterList
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName(std::string{kMaterialModelsName});
    tParameterList.sublist(std::string{kMaterialName}).set("Density", 0.5);
    return tParameterList;
}
[[nodiscard]] auto spatial_model() -> Teuchos::ParameterList
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName(std::string{kSpatialModelName});
    tParameterList.sublist("Domains").sublist("Design Volume").set("Element Block", "body");
    tParameterList.sublist("Domains").sublist("Design Volume").set("Material Model", std::string{kMaterialName});
    return tParameterList;
}

[[nodiscard]] auto criterionless_problem_for_test() -> Teuchos::ParameterList
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");
    tParameterList.sublist(std::string{kSpatialModelName}) = spatial_model();
    tParameterList.sublist(std::string{kMaterialModelsName}) = material_model();
    return tParameterList;
}

[[nodiscard]] auto mass_properties_criterion(const Teuchos::Array<std::string>& aPropertyList,
                                             const Teuchos::Array<double>& aWeightsList,
                                             const Teuchos::Array<double>& aGoldValuesList,
                                             const unsigned int aPower) -> Teuchos::ParameterList
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Criteria");
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Type", std::string{kMassPropertiesName});
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Properties", aPropertyList);
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Weights", aWeightsList);
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Gold Values", aGoldValuesList);
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Least Squares Exponent", aPower);
    return tParameterList;
}

[[nodiscard]] auto mass_property_problem(const Teuchos::Array<std::string>& aPropertyList,
                                         const Teuchos::Array<double>& aWeightsList,
                                         const Teuchos::Array<double>& aGoldValuesList,
                                         const unsigned int aPower = 2) -> Teuchos::ParameterList
{
    auto tParameterList = criterionless_problem_for_test();
    tParameterList.sublist("Criteria") =
        mass_properties_criterion(aPropertyList, aWeightsList, aGoldValuesList, aPower);
    return tParameterList;
}

[[nodiscard]] auto create_data_map_and_spatial_model(Plato::Mesh aMesh,
                                                     Teuchos::RCP<Teuchos::ParameterList> aParameterList)
    -> std::pair<Plato::DataMap, plato::domain::SpatialModel>
{
    Plato::DataMap tDataMap;
    const auto tParsedDomains = plato::domain::parse_domains(*aParameterList, aMesh);
    plato::domain::SpatialModel tSpatialModel(aMesh, tParsedDomains, tDataMap);
    return std::make_pair(tDataMap, tSpatialModel);
}

const auto tExpectThrowLambda =
    [](Teuchos::FancyOStream& out, bool& success, const Teuchos::RCP<Teuchos::ParameterList>& aParameterList)
{
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", /*MeshWidth*/ 1);
    auto [tDataMap, tSpatialModel] = create_data_map_and_spatial_model(tMesh, aParameterList);
    std::string tFuncName = "Mass Properties";
    TEST_THROW(Plato::Geometric::MassPropertiesFunction<Plato::Geometrical<Plato::Tet4>> tMassProperties(
                   tSpatialModel, tDataMap, *aParameterList, tFuncName);
               , std::runtime_error);
};

}  // namespace

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesParsingBadExponent)
{
    Teuchos::RCP<Teuchos::ParameterList> tParameterList =
        Teuchos::make_rcp<Teuchos::ParameterList>(mass_property_problem(
            /*PropertyList*/ {"Mass"}, /*WeightList*/ {2.0}, /*GoldList*/ {0.2}, /*Exponent*/ 3));
    tExpectThrowLambda(out, success, tParameterList);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesParsingGoldShouldBeZero)
{
    Teuchos::RCP<Teuchos::ParameterList> tParameterList =
        Teuchos::make_rcp<Teuchos::ParameterList>(mass_property_problem(
            /*PropertyList*/ {"Mass"}, /*WeightList*/ {2.0}, /*GoldList*/ {0.2}, /*Exponent*/ 1));
    tExpectThrowLambda(out, success, tParameterList);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassInsteadOfVolume2D)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    using Residual = typename Plato::Geometric::Evaluation<ElementType>::Residual;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    constexpr Plato::Scalar tPseudoDensity{0.8};
    Plato::blas1::fill(tPseudoDensity, tControl);

    Teuchos::RCP<Teuchos::ParameterList> tParameterList =
        Teuchos::make_rcp<Teuchos::ParameterList>(criterionless_problem_for_test());
    auto [tDataMap, tSpatialModel] = create_data_map_and_spatial_model(tMesh, tParameterList);
    Plato::Geometric::WeightedSumFunction<Plato::Geometrical<Plato::Tri3>> tWeightedSum(tSpatialModel, tDataMap);

    const auto tOnlyDomain = tSpatialModel.mDomains.front();
    const auto tCriterion = std::make_shared<Plato::Geometric::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    constexpr Plato::Scalar tMaterialDensity{0.5};
    tCriterion->setMaterialDensity(tMaterialDensity);
    tCriterion->setCalculationType("Mass");

    const auto tGeometryScalarFunc =
        std::make_shared<Plato::Geometric::GeometryScalarFunction<Plato::Geometrical<Plato::Tri3>>>(tSpatialModel,
                                                                                                    tDataMap);
    tGeometryScalarFunc->setEvaluator(tCriterion, tOnlyDomain.domainName());

    const Plato::Scalar tFunctionWeight = 0.75;
    tWeightedSum.allocateScalarFunctionBase(tGeometryScalarFunc);
    tWeightedSum.appendFunctionWeight(tFunctionWeight);
    const auto tObjFuncVal = tWeightedSum.value(tControl);

    Plato::Scalar tGoldValue =
        pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim) * tPseudoDensity * tFunctionWeight * tMaterialDensity;

    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, kTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassInsteadOfVolume3D)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    using Residual = typename Plato::Geometric::Evaluation<ElementType>::Residual;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    constexpr Plato::Scalar tPseudoDensity{0.8};
    Plato::blas1::fill(tPseudoDensity, tControl);

    Teuchos::RCP<Teuchos::ParameterList> tParameterList =
        Teuchos::make_rcp<Teuchos::ParameterList>(criterionless_problem_for_test());
    auto [tDataMap, tSpatialModel] = create_data_map_and_spatial_model(tMesh, tParameterList);
    Plato::Geometric::WeightedSumFunction<Plato::Geometrical<Plato::Tet4>> tWeightedSum(tSpatialModel, tDataMap);

    const auto tOnlyDomain = tSpatialModel.mDomains.front();
    const auto tCriterion = std::make_shared<Plato::Geometric::MassMoment<Residual>>(tOnlyDomain, tDataMap);
    constexpr Plato::Scalar tMaterialDensity{0.5};
    tCriterion->setMaterialDensity(tMaterialDensity);
    tCriterion->setCalculationType("Mass");

    const auto tGeometryScalarFunc =
        std::make_shared<Plato::Geometric::GeometryScalarFunction<Plato::Geometrical<Plato::Tet4>>>(tSpatialModel,
                                                                                                    tDataMap);
    tGeometryScalarFunc->setEvaluator(tCriterion, tOnlyDomain.domainName());
    tWeightedSum.allocateScalarFunctionBase(tGeometryScalarFunc);
    constexpr Plato::Scalar tFunctionWeight{0.75};
    tWeightedSum.appendFunctionWeight(tFunctionWeight);

    const auto tObjFuncVal = tWeightedSum.value(tControl);
    const Plato::Scalar tGoldValue =
        pow(static_cast<Plato::Scalar>(tMeshWidth), tSpaceDim) * tPseudoDensity * tFunctionWeight * tMaterialDensity;
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, kTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesValue3D)
{
    constexpr Plato::OrdinalType tMeshWidth{1};
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    constexpr Plato::Scalar tPseudoDensity{0.8};
    Plato::blas1::fill(tPseudoDensity, tControl);

    Teuchos::RCP<Teuchos::ParameterList> tParameterList =
        Teuchos::make_rcp<Teuchos::ParameterList>(mass_property_problem(
            /*PropertyList*/ {"Mass", "CGx", "CGy", "CGz", "Ixx", "Iyy", "Izz", "Ixy", "Iyz"},
            /*WeightList*/ {2.0, 0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
            /*GoldList*/ {0.2, 0.05, 0.55, 0.75, 0.5, 0.5, 0.5, 0.3, 0.3}));

    auto [tDataMap, tSpatialModel] = create_data_map_and_spatial_model(tMesh, tParameterList);
    std::string tFuncName = "Mass Properties";
    Plato::Geometric::MassPropertiesFunction<Plato::Geometrical<Plato::Tet4>> tMassProperties(
        tSpatialModel, tDataMap, *tParameterList, tFuncName);

    const auto tObjFuncVal = tMassProperties.value(tControl);
    const Plato::Scalar tGoldValue =
        2.0 * pow((0.4 - 0.2) / 0.2, 2) + 0.1 * pow((0.5 - 0.05), 2) + 2.0 * pow((0.5 - 0.55) / 0.55, 2) +
        3.0 * pow((0.5 - 0.75) / 0.75, 2) + 4.0 * pow((2.6666666666666666e-1 - 0.5) / 0.5, 2) +
        5.0 * pow((2.6666666666666666e-1 - 0.5) / 0.5, 2) + 6.0 * pow((2.6666666666666666e-1 - 0.5) / 0.5, 2) +
        7.0 * pow((-0.1 - 0.3) / 0.3, 2) + 8.0 * pow((-0.1 - 0.3) / 0.3, 2);
    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, kTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesValue3DNormalized)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    const Plato::OrdinalType tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Controls", tNumVerts);
    constexpr Plato::Scalar tPseudoDensity{0.8};
    Plato::blas1::fill(tPseudoDensity, tControl);

    Teuchos::RCP<Teuchos::ParameterList> tParameterList =
        Teuchos::make_rcp<Teuchos::ParameterList>(mass_property_problem(
            /*PropertyList*/ {"Mass", "CGx", "CGy", "CGz", "Ixx", "Iyy", "Izz", "Ixy", "Ixz", "Iyz"},
            /*WeightList*/ {2.0, 0.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0},
            /*GoldList*/ {0.2, 0.05, 0.55, 0.75, 5.4, 5.5, 5.4, -0.1, -0.1, -0.15}));

    auto [tDataMap, tSpatialModel] = create_data_map_and_spatial_model(tMesh, tParameterList);
    std::string tFuncName = "Mass Properties";
    Plato::Geometric::MassPropertiesFunction<Plato::Geometrical<Plato::Tet4>> tMassProperties(
        tSpatialModel, tDataMap, *tParameterList, tFuncName);

    const auto tObjFuncVal = tMassProperties.value(tControl);
    const Plato::Scalar tGoldValue = 2.0 * pow((0.4 - 0.2) / 0.2, 2) + 0.1 * pow((0.5 - 0.05), 2) +
                                     2.0 * pow((0.5 - 0.55) / 0.55, 2) + 3.0 * pow((0.5 - 0.75) / 0.75, 2) +
                                     4.0 * pow((-0.105801712354811 - 5.1240534614389617) / 5.1240534614389617, 2) +
                                     5.0 * pow((0.026312317550603 - 5.4403485162247298) / 5.4403485162247298, 2) +
                                     6.0 * pow((0.185489394804209 - 5.3885980223363132) / 5.3885980223363132, 2) +
                                     7.0 * pow((0.000176996782885 - 0.0000) / 5.1240534614389617, 2) +
                                     8.0 * pow((0.095340493277529 - 0.0000) / 5.1240534614389617, 2) +
                                     9.0 * pow((0.039658933738485 - 0.0000) / 5.1240534614389617, 2);

    TEST_FLOATING_EQUALITY(tGoldValue, tObjFuncVal, kTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, MassPropertiesGradZ_3D)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    using GradientZ = typename Plato::Geometric::Evaluation<ElementType>::GradientZ;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    Teuchos::RCP<Teuchos::ParameterList> tParameterList =
        Teuchos::make_rcp<Teuchos::ParameterList>(mass_property_problem(
            /*PropertyList*/ {"Mass", "CGx", "CGy", "CGz", "Ixx", "Iyy", "Izz", "Ixy", "Iyz"},
            /*WeightList*/ {2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
            /*GoldList*/ {0.2, 0.45, 0.55, 0.75, 0.5, 0.5, 0.5, 0.3, 0.3}));

    auto [tDataMap, tSpatialModel] = create_data_map_and_spatial_model(tMesh, tParameterList);
    std::string tFuncName = "Mass Properties";
    Plato::Geometric::MassPropertiesFunction<Plato::Geometrical<Plato::Tet4>> tMassProperties(
        tSpatialModel, tDataMap, *tParameterList, tFuncName);
    Plato::test_partial_control<GradientZ, ElementType>(tMesh, tMassProperties);
}

}  // namespace problem::geometric::unittest
