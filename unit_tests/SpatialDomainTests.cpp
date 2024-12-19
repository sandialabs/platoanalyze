#include <Teuchos_XMLParameterListHelpers.hpp>
#include <filesystem>
#include <string_view>

#include "PlatoMesh.hpp"
#include "SpatialModel.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include "util/PlatoMeshTestHelpers.hpp"

namespace PlatoUnitTests {
namespace {
const auto kMeshFilePath = std::filesystem::path{"test-mesh.exo"};

auto write_and_load_mesh() -> Plato::Mesh {
  Plato::TestHelpers::write_two_block_mesh(kMeshFilePath);
  return Plato::MeshFactory::create(kMeshFilePath.string());
}

class TwoBlockMeshRAII {
 public:
  Plato::Mesh mMesh = write_and_load_mesh();

  ~TwoBlockMeshRAII() { std::filesystem::remove(kMeshFilePath); }
};
}  // namespace

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialModelIgnoreMissingElementBlocks) {
  const auto tTestInput = [](const bool aParameterValue) {
    const auto tParameterAsString = std::string{aParameterValue ? "true" : "false"};
    const auto tInput = std::string{
        "<ParameterList name='Spatial Model'>\n"
        "  <Parameter name='Ignore Missing Element Blocks' type='bool' value='" +
        tParameterAsString +
        "'/>"
        "</ParameterList>\n"};
    return Teuchos::getParametersFromXmlString(tInput);
  };

  TEST_ASSERT(Plato::SpatialModel::ignoreMissingElementBlocks(*tTestInput(true)));
  TEST_ASSERT(!Plato::SpatialModel::ignoreMissingElementBlocks(*tTestInput(false)));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialModelIgnoreMissingElementBlocksDefault) {
  const auto tInput = std::string{
      "<ParameterList name='Spatial Model'>\n"
      "  <Parameter name='Ignore Missing Element Blocks Wrong name' type='bool' value='false'/>"
      "</ParameterList>\n"};
  const auto tParameterList = Teuchos::getParametersFromXmlString(tInput);

  TEST_ASSERT(!Plato::SpatialModel::ignoreMissingElementBlocks(*tParameterList));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainElementBlockName) {
  constexpr auto tInput =
      "<ParameterList name='Design Volume'>\n"
      "  <Parameter name='Element Block' type='string' value='octopus'/>\n"
      "</ParameterList>\n";
  const auto tParameterList = Teuchos::getParametersFromXmlString(tInput);
  const auto tBlockName = Plato::SpatialDomain::elementBlockName(*tParameterList);
  TEST_ASSERT(tBlockName.has_value());
  TEST_EQUALITY(tBlockName.value(), "octopus");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainElementBlockNameDoesNotExist) {
  constexpr auto tInput =
      "<ParameterList name='Design Volume'>\n"
      "  <Parameter name='Element Bloke' type='string' value='squid'/>\n"
      "</ParameterList>\n";
  const auto tParameterList = Teuchos::getParametersFromXmlString(tInput);
  TEST_ASSERT(!Plato::SpatialDomain::elementBlockName(*tParameterList).has_value());
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainMeshHasElementBlock) {
  const auto tMesh = TwoBlockMeshRAII{};
  constexpr auto tInput =
      "<ParameterList name='Design Volume'>\n"
      "  <Parameter name='Element Block' type='string' value='BLOCK_1'/>\n"
      "</ParameterList>\n";
  const auto tParameterList = Teuchos::getParametersFromXmlString(tInput);
  TEST_ASSERT(Plato::SpatialDomain::elementBlockExistsInMesh(tMesh.mMesh, *tParameterList));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialDomainMeshWrongElementBlockTag) {
  const auto tMesh = TwoBlockMeshRAII{};
  constexpr auto tInput =
      "<ParameterList name='Design Volume'>\n"
      "  <Parameter name='Element Bloke' type='string' value='BLOCK_1'/>\n"
      "</ParameterList>\n";
  const auto tParameterList = Teuchos::getParametersFromXmlString(tInput);
  TEST_ASSERT(!Plato::SpatialDomain::elementBlockExistsInMesh(tMesh.mMesh, *tParameterList));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SpatialModelIgnoresMissingBlocks) {
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
