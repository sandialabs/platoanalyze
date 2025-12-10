#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <string_view>

#include "domain/SpatialModel.hpp"
#include "mesh/PlatoMesh.hpp"
#include "test_utilities/PlatoMeshTestHelpers.hpp"

namespace plato::domain::unittest
{
namespace
{
auto domain_parameter_list(const std::string_view aElementBlockParameterTag, const std::string_view aElementBlockName)
    -> Teuchos::ParameterList
{
    auto tParameterList = Teuchos::ParameterList{};
    tParameterList.set(std::string{aElementBlockParameterTag}, std::string{aElementBlockName});
    return tParameterList;
}

auto spatial_model_with_ignore_mismatch_parameter(const std::string_view aIgnoreMismatchParameterTag, const bool aValue)
    -> Teuchos::ParameterList
{
    auto tParameterList = Teuchos::ParameterList{};
    tParameterList.set(std::string{aIgnoreMismatchParameterTag}, aValue);
    return tParameterList;
}
}  // namespace

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialModelIgnoreMissingElementBlocks)
{
    constexpr auto tIgnoreMismatchParameterName = std::string_view{"Ignore Missing Element Blocks"};
    TEST_ASSERT(ignore_missing_element_blocks(
        spatial_model_with_ignore_mismatch_parameter(tIgnoreMismatchParameterName, true)));
    TEST_ASSERT(!ignore_missing_element_blocks(
        spatial_model_with_ignore_mismatch_parameter(tIgnoreMismatchParameterName, false)));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialModelIgnoreMissingElementBlocksDefault)
{
    TEST_ASSERT(!ignore_missing_element_blocks(
        spatial_model_with_ignore_mismatch_parameter("Ignore Missing Element Blocks Wrong name", false)));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainElementBlockName)
{
    const auto tParameterList = domain_parameter_list("Element Block", "octopus");
    const auto tBlockName = detail::element_block_name(tParameterList);
    TEST_ASSERT(tBlockName.has_value());
    TEST_EQUALITY(tBlockName.value(), "octopus");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainElementBlockNameDoesNotExist)
{
    const auto tParameterList = domain_parameter_list("TYPO Element Bloke", "squid");
    TEST_ASSERT(!detail::element_block_name(tParameterList).has_value());
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainMeshHasElementBlock)
{
    const auto tMesh = Plato::TestHelpers::TwoBlockTriMeshRAII{};
    const auto tBlockName = std::string{"BLOCK_1"};
    TEST_ASSERT(detail::element_block_exists_in_mesh(tMesh.mMesh, tBlockName));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainStdOptional)
{
    const auto tMesh = Plato::TestHelpers::TwoBlockTriMeshRAII{};
    const std::optional<std::string> tBlockName = std::nullopt;
    TEST_ASSERT(!detail::element_block_exists_in_mesh(tMesh.mMesh, tBlockName));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainMeshWrongElementBlockTag)
{
    const auto tMesh = Plato::TestHelpers::TwoBlockTriMeshRAII{};
    const auto tBlockName = std::string{"TYPO_BLOCK_1"};
    TEST_ASSERT(!detail::element_block_exists_in_mesh(tMesh.mMesh, tBlockName));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialModelIgnoresMissingBlocks)
{
    const auto tMesh = Plato::TestHelpers::TwoBlockTriMeshRAII{};

    const auto tInput = std::string{
        "<ParameterList name='Plato Problem'>\n"
        "<ParameterList name='Spatial Model'>\n"
        "  <Parameter name='Ignore Missing Element Blocks' type='bool' value='true'/>"
        "  <ParameterList name='Domains'>\n"
        "    <ParameterList name='Design Volume'>\n"
        "      <Parameter name='Element Block' type='string' value='BLOCK_1'/>\n"
        "      <Parameter name='Material Model' type='string' value='Fancy Feast'/>\n"
        "    </ParameterList>\n"
        "    <ParameterList name='Void Volume'>\n"
        "      <Parameter name='Element Block' type='string' value='BLOCK_1_void'/>\n"
        "      <Parameter name='Material Model' type='string' value='Unfancy Feast'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n"
        "</ParameterList>\n"
        "</ParameterList>\n"};
    const auto tParameterList = Teuchos::getParametersFromXmlString(tInput);
    auto tDataMap = Plato::DataMap{};
    const auto tParsedDomains = plato::domain::parse_domains(*tParameterList, tMesh.mMesh);
    const auto tSpatialModel = plato::domain::SpatialModel{tMesh.mMesh, tParsedDomains, tDataMap};

    constexpr auto tExpectedNumberOfDomains = 1U;
    TEST_EQUALITY_CONST(tSpatialModel.mDomains.size(), tExpectedNumberOfDomains);

    tParameterList->sublist("Spatial Model").get<bool>("Ignore Missing Element Blocks") = false;
    TEST_THROW(const auto tParsedDomains = plato::domain::parse_domains(*tParameterList, tMesh.mMesh);
               , std::runtime_error);
}

}  // namespace plato::domain::unittest
