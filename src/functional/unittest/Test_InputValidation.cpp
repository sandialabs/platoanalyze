#include <Teuchos_UnitTestHarness.hpp>
#include <plato/filter/library/FilterInterface.hpp>

#include "FunctionalInterfaceUtilities.hpp"
#include "InputValidation.hpp"
#include "PlatoTestHelpers.hpp"
#include "TestMeshSetupTeardown.hpp"

namespace plato::functional::unittest
{
namespace
{
constexpr auto kMeshWidth = int{1};
constexpr auto kMeshName = std::string_view{"not-a-mesh.exo"};

/// @brief RAII class for creating and destroying a box mesh for testing.
struct BoxMeshFixture
{
    BoxMeshFixture() : mMesh{Plato::TestHelpers::get_box_mesh("TET4", kMeshWidth)} {}
    ~BoxMeshFixture() { std::filesystem::remove(mMesh->FileName()); }

    Plato::Mesh mMesh;
};

constexpr auto kExpectSuccess = true;
constexpr auto kExpectFailure = false;

void test_affirm_mesh_blocks_match_input(const Plato::Mesh& aMesh,
                                         const std::vector<std::string>& aBlockNames,
                                         const bool aExpectSuccess,
                                         Teuchos::FancyOStream& aOutStream,
                                         bool& aSuccess)
{
    const auto tParameterList =
        helmholtz_filter_parameter_list(filter::library::FilterParameters{}, kMeshName, aBlockNames);
    TEUCHOS_TEST_EQUALITY(aExpectSuccess, affirm_mesh_blocks_match_input(tParameterList, aMesh), aOutStream, aSuccess);
    if (aExpectSuccess != affirm_mesh_blocks_match_input(tParameterList, aMesh))
    {
        aOutStream << error_messages(tParameterList, aMesh) << "\n";
    }
}
}  // namespace

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AffirmInputMeshBlocksMatchMeshAllValid)
{
    // 1 block
    {
        const auto tBoxMeshFixture = BoxMeshFixture{};
        const auto tBlocks = tBoxMeshFixture.mMesh->GetElementBlockNames();
        test_affirm_mesh_blocks_match_input(tBoxMeshFixture.mMesh, tBlocks, kExpectSuccess, out, success);
    }
    // 2 blocks
    {
        const auto tTestFixture = TestMeshSetupTeardown{};
        const auto tBlocks = tTestFixture.mesh()->GetElementBlockNames();
        test_affirm_mesh_blocks_match_input(tTestFixture.mesh(), tBlocks, kExpectSuccess, out, success);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AffirmInputMeshBlocksMatchMeshAllInvalid)
{
    const auto tBoxMeshFixture = BoxMeshFixture{};
    const auto tBlocks = std::vector<std::string>{"allosaurus"};
    test_affirm_mesh_blocks_match_input(tBoxMeshFixture.mMesh, tBlocks, kExpectFailure, out, success);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AffirmInputMeshBlocksMatchMeshBothValidAndInvalid)
{
    // 2 blocks in input, 1 block in mesh
    {
        const auto tBoxMeshFixture = BoxMeshFixture{};
        const auto tValidBlocks = tBoxMeshFixture.mMesh->GetElementBlockNames();
        constexpr auto tExpectedNumberOfBlocks = 1U;
        TEST_EQUALITY(tValidBlocks.size(), tExpectedNumberOfBlocks);
        const auto tBlocks = std::vector<std::string>{"deinosuchus", tValidBlocks.front()};
        test_affirm_mesh_blocks_match_input(tBoxMeshFixture.mMesh, tBlocks, kExpectSuccess, out, success);
    }
    // 2 blocks in input, 2 in mesh
    {
        const auto tTestFixture = TestMeshSetupTeardown{};
        const auto tValidBlocks = tTestFixture.mesh()->GetElementBlockNames();
        constexpr auto tExpectedNumberOfBlocks = 2U;
        TEST_EQUALITY(tValidBlocks.size(), tExpectedNumberOfBlocks);
        const auto tBlocks = std::vector<std::string>{"spinosaurus", tValidBlocks.front()};
        test_affirm_mesh_blocks_match_input(tTestFixture.mesh(), tBlocks, kExpectFailure, out, success);
    }
    // 3 blocks in input, 1 in mesh
    {
        const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", kMeshWidth);
        const auto tValidBlocks = tMesh->GetElementBlockNames();
        constexpr auto tExpectedNumberOfBlocks = 1U;
        TEST_EQUALITY(tValidBlocks.size(), tExpectedNumberOfBlocks);
        const auto tBlocks = std::vector<std::string>{tValidBlocks.front(), tValidBlocks.front(), "diplodocus"};
        test_affirm_mesh_blocks_match_input(tMesh, tBlocks, kExpectSuccess, out, success);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AffirmInputMeshBlocksMatchMeshRepeatedBlocks)
{
    const auto tTestFixture = TestMeshSetupTeardown{};
    const auto tValidBlocks = tTestFixture.mesh()->GetElementBlockNames();
    constexpr auto tExpectedSize = 2U;
    TEST_EQUALITY(tValidBlocks.size(), tExpectedSize);
    const auto tBlocksForParameterList = std::vector<std::string>{tValidBlocks.front(), tValidBlocks.front()};
    test_affirm_mesh_blocks_match_input(tTestFixture.mesh(), tBlocksForParameterList, kExpectFailure, out, success);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ErrorMessages)
{
    const auto tTestFixture = TestMeshSetupTeardown{};
    const auto tValidBlocks = tTestFixture.mesh()->GetElementBlockNames();
    // Erroneous
    {
        const auto tBlocksForParameterList = std::vector<std::string>{tValidBlocks.front(), tValidBlocks.front()};
        const auto tParameterList =
            helmholtz_filter_parameter_list(filter::library::FilterParameters{}, kMeshName, tBlocksForParameterList);
        const auto tErrorMessage = error_messages(tParameterList, tTestFixture.mesh());
        TEST_ASSERT(!tErrorMessage.empty());
        // Check that each block name appears in the error message
        TEST_INEQUALITY(tErrorMessage.find(tValidBlocks.front()), std::string::npos);
        TEST_INEQUALITY(tErrorMessage.find(tValidBlocks.back()), std::string::npos);
    }
    // Valid
    {
        const auto tBlocksForParameterList = tValidBlocks;
        const auto tParameterList =
            helmholtz_filter_parameter_list(filter::library::FilterParameters{}, kMeshName, tBlocksForParameterList);
        const auto tErrorMessage = error_messages(tParameterList, tTestFixture.mesh());
        TEST_EQUALITY_CONST(tErrorMessage, "");
    }
}

}  // namespace plato::functional::unittest
