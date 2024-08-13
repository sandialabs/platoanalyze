#include <Teuchos_UnitTestHarness.hpp>

#include "EngineMesh.hpp"
#include "PlatoMesh.hpp"
#include "util/PlatoMeshTestHelpers.hpp"

namespace Plato::UnitTests {
TEUCHOS_UNIT_TEST(EngineMeshTests, NodeMapIdentity) {
  const auto tMeshFilePath = std::filesystem::path{"test-mesh.exo"};
  TestHelpers::write_one_block_mesh(tMeshFilePath);
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

}  // namespace Plato::UnitTests
