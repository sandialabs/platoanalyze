#include <Teuchos_XMLParameterListHelpers.hpp>
#include <filesystem>
#include <string_view>

#include "PlatoMesh.hpp"
#include "SpatialModel.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "util/PlatoMeshTestHelpers.hpp"

namespace PlatoUnitTests
{
namespace
{
const auto kMeshFilePath = std::filesystem::path{"test-mesh.exo"};

auto write_and_load_mesh() -> Plato::Mesh
{
    Plato::TestHelpers::write_two_block_mesh(kMeshFilePath);
    return Plato::MeshFactory::create(kMeshFilePath.string());
}

class TwoBlockMeshRAII
{
   public:
    Plato::Mesh mMesh = write_and_load_mesh();

    ~TwoBlockMeshRAII() { std::filesystem::remove(kMeshFilePath); }
};

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
    TEST_ASSERT(Plato::SpatialModel::ignoreMissingElementBlocks(
        spatial_model_with_ignore_mismatch_parameter(tIgnoreMismatchParameterName, true)));
    TEST_ASSERT(!Plato::SpatialModel::ignoreMissingElementBlocks(
        spatial_model_with_ignore_mismatch_parameter(tIgnoreMismatchParameterName, false)));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialModelIgnoreMissingElementBlocksDefault)
{
    TEST_ASSERT(!Plato::SpatialModel::ignoreMissingElementBlocks(
        spatial_model_with_ignore_mismatch_parameter("Ignore Missing Element Blocks Wrong name", false)));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainElementBlockName)
{
    const auto tParameterList = domain_parameter_list("Element Block", "octopus");
    const auto tBlockName = Plato::SpatialDomain::elementBlockName(tParameterList);
    TEST_ASSERT(tBlockName.has_value());
    TEST_EQUALITY(tBlockName.value(), "octopus");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainElementBlockNameDoesNotExist)
{
    const auto tParameterList = domain_parameter_list("Element Bloke", "squid");
    TEST_ASSERT(!Plato::SpatialDomain::elementBlockName(tParameterList).has_value());
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainMeshHasElementBlock)
{
    const auto tMesh = TwoBlockMeshRAII{};
    const auto tParameterList = domain_parameter_list("Element Block", "BLOCK_1");
    TEST_ASSERT(Plato::SpatialDomain::elementBlockExistsInMesh(tMesh.mMesh, tParameterList));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainMeshWrongElementBlockTag)
{
    const auto tMesh = TwoBlockMeshRAII{};
    const auto tParameterList = domain_parameter_list("Element Bloke", "BLOCK_1");
    TEST_ASSERT(!Plato::SpatialDomain::elementBlockExistsInMesh(tMesh.mMesh, tParameterList));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialModelIgnoresMissingBlocks)
{
    const auto tMesh = TwoBlockMeshRAII{};

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
    const auto tSpatialModel = Plato::SpatialModel{tMesh.mMesh, *tParameterList, tDataMap};

    constexpr auto tExpectedNumberOfDomains = 1U;
    TEST_EQUALITY_CONST(tSpatialModel.Domains.size(), tExpectedNumberOfDomains);

    tParameterList->sublist("Spatial Model").get<bool>("Ignore Missing Element Blocks") = false;
    TEST_THROW(Plato::SpatialModel(tMesh.mMesh, *tParameterList, tDataMap), std::runtime_error);
}

}  // namespace PlatoUnitTests
