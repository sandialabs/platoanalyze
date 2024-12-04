#include <Teuchos_UnitTestHarness.hpp>
#include <plato/filter/FilterInterface.hpp>

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

void test_affirm_input_mesh_blocks_match_mesh(const Plato::Mesh& aMesh,
                                              const std::vector<std::string>& aBlockNames,
                                              const bool aExpectSuccess,
                                              Teuchos::FancyOStream& aOutStream,
                                              bool& aSuccess)
{
    const auto tParameterList =
        helmholtz_filter_parameter_list(filter::library::FilterParameters{}, kMeshName, aBlockNames);
    TEUCHOS_TEST_EQUALITY(aExpectSuccess, affirm_input_mesh_blocks_match_mesh(tParameterList, aMesh), aOutStream,
                          aSuccess);
}
}  // namespace

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AffirmInputMeshBlocksMatchMeshAllValid)
{
    // 1 block
    {
        const auto tBoxMeshFixture = BoxMeshFixture{};
        const auto tBlocks = tBoxMeshFixture.mMesh->GetElementBlockNames();
        test_affirm_input_mesh_blocks_match_mesh(tBoxMeshFixture.mMesh, tBlocks, kExpectSuccess, out, success);
    }
    // 2 blocks
    {
        const auto tTestFixture = TestMeshSetupTeardown{};
        const auto tBlocks = tTestFixture.mesh()->GetElementBlockNames();
        test_affirm_input_mesh_blocks_match_mesh(tTestFixture.mesh(), tBlocks, kExpectSuccess, out, success);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AffirmInputMeshBlocksMatchMeshAllInvalid)
{
    const auto tBoxMeshFixture = BoxMeshFixture{};
    const auto tBlocks = std::vector<std::string>{"allosaurus"};
    test_affirm_input_mesh_blocks_match_mesh(tBoxMeshFixture.mMesh, tBlocks, kExpectFailure, out, success);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AffirmInputMeshBlocksMatchMeshBothValidAndInvalid)
{
    // 1 block
    {
        const auto tBoxMeshFixture = BoxMeshFixture{};
        const auto tValidBlocks = tBoxMeshFixture.mMesh->GetElementBlockNames();
        const auto tBlocks = std::vector<std::string>{"deinosuchus", tValidBlocks.front()};
        test_affirm_input_mesh_blocks_match_mesh(tBoxMeshFixture.mMesh, tBlocks, kExpectFailure, out, success);
    }
    // 2 blocks, 1 valid, 1 invalid
    {
        const auto tTestFixture = TestMeshSetupTeardown{};
        const auto tValidBlocks = tTestFixture.mesh()->GetElementBlockNames();
        const auto tBlocks = std::vector<std::string>{"spinosaurus", tValidBlocks.front()};
        test_affirm_input_mesh_blocks_match_mesh(tTestFixture.mesh(), tBlocks, kExpectFailure, out, success);
    }
    // 3 blocks, 2 valid, 1 invalid
    {
        const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", kMeshWidth);
        const auto tValidBlocks = tMesh->GetElementBlockNames();
        const auto tBlocks = std::vector<std::string>{tValidBlocks.front(), tValidBlocks.back(), "diplodocus"};
        test_affirm_input_mesh_blocks_match_mesh(tMesh, tBlocks, kExpectFailure, out, success);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AffirmInputMeshBlocksMatchMeshRepeatedBlocks)
{
    const auto tTestFixture = TestMeshSetupTeardown{};
    const auto tValidBlocks = tTestFixture.mesh()->GetElementBlockNames();
    constexpr auto tExpectedSize = 2U;
    TEST_EQUALITY(tValidBlocks.size(), tExpectedSize);
    const auto tBlocksForParameterList = std::vector<std::string>{tValidBlocks.front(), tValidBlocks.front()};
    test_affirm_input_mesh_blocks_match_mesh(tTestFixture.mesh(), tBlocksForParameterList, kExpectFailure, out,
                                             success);
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
