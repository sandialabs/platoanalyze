#include <Teuchos_UnitTestHarness.hpp>

#include "EngineMesh.hpp"
#include "PlatoMesh.hpp"
#include "util/PlatoMeshTestHelpers.hpp"

namespace Plato::UnitTests {
TEUCHOS_UNIT_TEST(EngineMeshTests, NodeMapIdentity) {
  const auto tMeshFilePath = std::filesystem::path{"test-mesh.exo"};
  TestHelpers::write_mesh_with_nodemap(tMeshFilePath);
  const auto tMesh = Plato::MeshFactory::create(tMeshFilePath.string());

  const auto tNodeMap = tMesh->NodeMap();

  TEST_EQUALITY(tNodeMap.size(), tMesh->NumNodes());
  for (const auto [tKey, tValue] : tNodeMap) {
    // A 1-to-1 node map is indexed from 1 (not 0), so these should be 1 apart
    TEST_EQUALITY(tKey, tValue + 1);
  }

  std::filesystem::remove(tMeshFilePath);
}

TEUCHOS_UNIT_TEST(EngineMeshTests, NodeMapArbitrary) {
  // Values copied from the original node map
  const auto tExpectedNodeMap = std::vector<unsigned>{31, 34, 35, 30, 28, 33, 32, 29};
  constexpr auto tMeshFileName = "hex-1-1-1.exo";
  const auto tMesh = Plato::MeshFactory::create(tMeshFileName);

  const auto tNodeMap = tMesh->NodeMap();

  TEST_EQUALITY(tNodeMap.size(), tMesh->NumNodes());
  for (const auto [tKey, tValue] : tNodeMap) {
    TEST_EQUALITY(tKey, tExpectedNodeMap.at(tValue));
  }
}

TEUCHOS_UNIT_TEST(EngineMeshTest, BlockIDMapTwoBlockMesh) {
  constexpr auto tMeshFileName = "two_block_contact.exo";
  const auto tMesh = Plato::MeshFactory::create(tMeshFileName);

  const auto tBlockIDMap = tMesh->BlockIDMap();

  constexpr auto tExpectedMapSize = 2U;
  TEST_EQUALITY(tBlockIDMap.size(), tExpectedMapSize);

  const auto tExpectedBlockIDs = std::vector{1U, 2U};
  const auto tExpectedBlockNames = std::vector<std::string_view>{"block_1", "block_2"};
  auto tBlockIDIterator = tBlockIDMap.cbegin();
  for (auto tIndex = 0U; tIndex < tExpectedBlockIDs.size(); ++tIndex) {
    TEST_EQUALITY(tBlockIDIterator->first, tExpectedBlockIDs[tIndex]);
    TEST_EQUALITY(tBlockIDIterator->second, tExpectedBlockNames[tIndex]);
    ++tBlockIDIterator;
  }
}

}  // namespace Plato::UnitTests
