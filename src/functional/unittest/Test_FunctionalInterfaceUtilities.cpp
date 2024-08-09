#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <chrono>
#include <numeric>
#include <plato/filter/FilterInterface.hpp>
#include <random>

#include "BLAS1.hpp"
#include "FunctionalInterfaceUtilities.hpp"
#include "PlatoMeshTestHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTestHelpers.hpp"

namespace plato::functional::unittest
{
namespace
{
// A vector of nodal densities with densities equal to the global ID. This can be used for tests that include all blocks
// (none fixed).
const auto tDensityVector1AllBlocks =
    std::vector<plato::mesh::Density>{{1, 0, 1.0}, {2, 1, 2.0}, {3, 2, 3.0}, {4, 3, 4.0}, {5, 4, 5.0}};
const auto tDensityVector2AllBlocks =
    std::vector<plato::mesh::Density>{{3, 2, 3.0}, {4, 3, 4.0}, {5, 4, 5.0}, {6, 5, 6.0}, {7, 6, 7.0}, {8, 7, 8.0}};

// A vector of nodal densities for block 1 with densities equal to the vector index assuming block 2 is fixed.
// The densities are set to the vector index.
const auto tDensityVector1Block2Fixed =
    std::vector<plato::mesh::Density>{{1, 0, 1.0}, {2, 1, 2.0}, {3, 2, 3.0}, {4, 3, 4.0}, {5, 4, 5.0}};

// A vector of nodal densities for block 2 with densities equal to the vector index assuming block 1 is fixed.
// The densities are set to the vector index.
const auto tDensityVector2Block1Fixed =
    std::vector<plato::mesh::Density>{{3, 0, 3.0}, {4, 1, 4.0}, {5, 2, 5.0}, {6, 3, 6.0}, {7, 4, 7.0}, {8, 5, 8.0}};

const auto tVectorDensities = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

std::vector<double> density_values(const std::vector<plato::mesh::Density>& aDensityVector)
{
    auto tDensityValues = std::vector<double>{};
    tDensityValues.reserve(aDensityVector.size());
    std::transform(aDensityVector.cbegin(), aDensityVector.cend(), std::back_inserter(tDensityValues),
                   [](const auto& tDensity) { return tDensity.mDensity; });
    return tDensityValues;
}

Plato::Mesh test_mesh(const std::filesystem::path& aMeshFilePath)
{
    Plato::TestHelpers::write_two_block_mesh(aMeshFilePath);
    return Plato::MeshFactory::create(aMeshFilePath.string());
}

class TestMeshSetupTeardown
{
   public:
    ~TestMeshSetupTeardown() { std::filesystem::remove(mTestMeshPath); }

    auto meshDesignVariablesAllDesignBlocks() const -> plato::mesh::MeshDesignVariables
    {
        return plato::mesh::MeshDesignVariables{mTestMeshPath,
                                                {{1, tDensityVector1AllBlocks}, {2, tDensityVector2AllBlocks}}};
    }

    auto meshDesignVariablesBlock1Fixed() const -> plato::mesh::MeshDesignVariables
    {
        return plato::mesh::MeshDesignVariables{mTestMeshPath, {{2, tDensityVector2Block1Fixed}}};
    }

    auto meshDesignVariablesBlock2Fixed() const -> plato::mesh::MeshDesignVariables
    {
        return plato::mesh::MeshDesignVariables{mTestMeshPath, {{1, tDensityVector1Block2Fixed}}};
    }

    const Plato::Mesh& mesh() const { return mMesh; }

   private:
    std::filesystem::path mTestMeshPath = "test-mesh.exo";
    Plato::Mesh mMesh = test_mesh(mTestMeshPath);
};

template <typename TransformFunction>
void fill_with_transformed_indices(const Plato::ScalarVector tVectorOnDevice,
                                   const TransformFunction& aTransformFunction)
{
    const auto tVectorOnHost = Kokkos::create_mirror_view(tVectorOnDevice);
    for (auto tIndex = 0U; tIndex < tVectorOnHost.size(); ++tIndex)
    {
        tVectorOnHost[tIndex] = aTransformFunction(tIndex);
    }
    Kokkos::deep_copy(tVectorOnDevice, tVectorOnHost);
}

}  // namespace

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ParameterList)
{
    constexpr double tFilterRadius = 42.0;
    constexpr double tBoundaryStickingPenalty = 13.0;
    const auto tFilterParameters =
        filter::library::FilterParameters{/*.mFilterRadius=*/tFilterRadius,
                                          /*.mBoundaryStickingPenalty=*/tBoundaryStickingPenalty};
    constexpr auto tMeshName = std::string_view{"not-a-mesh.exo"};
    const Teuchos::ParameterList tParameterList = helmholtz_filter_parameter_list(tFilterParameters, tMeshName);

    TEST_EQUALITY(tParameterList.get<std::string>("Physics"), "Plato Driver");
    TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters").get<double>("Length Scale"),
                  tFilterRadius);
    TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters").get<double>("Surface Length Scale"),
                  tBoundaryStickingPenalty);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, UpdateMesh)
{
    constexpr std::string_view tInitialMeshName = "first-mesh-name.exo";
    Teuchos::ParameterList tParameterList =
        helmholtz_filter_parameter_list(filter::library::FilterParameters{}, tInitialMeshName);

    TEST_EQUALITY(tParameterList.get<std::string>("Input Mesh"), std::string{tInitialMeshName});

    constexpr std::string_view tNewMeshName = "second-mesh-name.exo";
    update_mesh_file_name(tParameterList, tNewMeshName);
    TEST_EQUALITY(tParameterList.get<std::string>("Input Mesh"), std::string{tNewMeshName});
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, HashCurrentDesign_MeshChanges)
{
    constexpr int tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    constexpr unsigned int tSize = 10;
    const auto tControl = Plato::ScalarVector("test", tSize);
    Plato::blas1::fill(1.0, tControl);
    auto tOriginalHash = hash_current_design(tControl, tMesh);

    auto tNewHash = hash_current_design(tControl, tMesh);
    TEST_EQUALITY(tNewHash, tOriginalHash);

    constexpr int tNewMeshWidth = 2;
    tMesh = Plato::TestHelpers::get_box_mesh("TET4", tNewMeshWidth);
    tNewHash = hash_current_design(tControl, tMesh);
    TEST_INEQUALITY(tNewHash, tOriginalHash);

    // regenerate the original mesh and ensure that the hash matches original
    tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    tNewHash = hash_current_design(tControl, tMesh);
    TEST_EQUALITY(tNewHash, tOriginalHash);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, HashCurrentDesign_ControlChanges)
{
    constexpr int tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    constexpr unsigned int tSize = 10;
    const auto tControl = Plato::ScalarVector("test", tSize);
    Plato::blas1::fill(1.0, tControl);
    auto tOriginalHash = hash_current_design(tControl, tMesh);

    auto tNewHash = hash_current_design(tControl, tMesh);
    TEST_EQUALITY(tNewHash, tOriginalHash);

    Plato::blas1::fill(0.5, tControl);
    tNewHash = hash_current_design(tControl, tMesh);
    TEST_INEQUALITY(tNewHash, tOriginalHash);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, DesignVariableStdVectorAllBlocks)
{
    const auto tTestFixture = TestMeshSetupTeardown{};

    const auto tNumEntries = tTestFixture.mesh()->NumNodes();
    const auto tControl = Plato::ScalarVector("test", tNumEntries);
    fill_with_transformed_indices(tControl, [](const auto tIndex) { return static_cast<double>(tIndex); });

    const auto tResult =
        design_variable_std_vector(tControl, tTestFixture.meshDesignVariablesAllDesignBlocks(), tTestFixture.mesh());
    const auto tExpected = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tControl);
    TEST_EQUALITY(tResult.size(), tExpected.size());
    for (std::size_t tIndex = 0; tIndex < tExpected.size(); ++tIndex)
    {
        TEST_EQUALITY(tResult.at(tIndex), tExpected[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, DesignVariableStdVectorFixedBlock)
{
    const auto tTestFixture = TestMeshSetupTeardown{};

    const auto tNumEntries = tTestFixture.mesh()->NumNodes();
    const auto tControl = Plato::ScalarVector("test", tNumEntries);
    fill_with_transformed_indices(tControl, [](const auto tIndex) { return static_cast<double>(tIndex); });

    const auto tResult =
        design_variable_std_vector(tControl, tTestFixture.meshDesignVariablesBlock1Fixed(), tTestFixture.mesh());

    constexpr auto tNumberOfFixedNodes = 2U;
    const auto tExpectedSize = tControl.size() - tNumberOfFixedNodes;
    TEST_EQUALITY(tResult.size(), tExpectedSize);
    for (std::size_t tIndex = tNumberOfFixedNodes; tIndex < tControl.size(); ++tIndex)
    {
        TEST_EQUALITY(tResult.at(tIndex - tNumberOfFixedNodes), tControl(tIndex));
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, FullNodalScalarVectorFromVectorAllBlocks)
{
    const auto tTestFixture = TestMeshSetupTeardown{};
    const auto tVector = std::vector<double>(tTestFixture.mesh()->NumNodes(), 42.0);

    const auto tMeshDesignVariables = tTestFixture.meshDesignVariablesAllDesignBlocks();

    const auto tResult = full_nodal_scalar_vector(tVector, tMeshDesignVariables, tTestFixture.mesh());
    const auto tResultOnHost = Kokkos::create_mirror_view(tResult);
    Kokkos::deep_copy(tResultOnHost, tResult);

    TEST_EQUALITY(tResultOnHost.size(), tVector.size());
    for (std::size_t tIndex = 0; tIndex < tVector.size(); ++tIndex)
    {
        TEST_EQUALITY(tResultOnHost[tIndex], tVector[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, FullNodalScalarVectorFromVectorFixedBlock1)
{
    const auto tTestFixture = TestMeshSetupTeardown{};

    const auto tVector = density_values(tDensityVector2Block1Fixed);
    const auto tMeshDesignVariables = tTestFixture.meshDesignVariablesBlock1Fixed();

    const auto tResult = full_nodal_scalar_vector(tVector, tMeshDesignVariables, tTestFixture.mesh());
    const auto tResultOnHost = Kokkos::create_mirror_view(tResult);
    Kokkos::deep_copy(tResultOnHost, tResult);

    TEST_EQUALITY(tResultOnHost.size(), tTestFixture.mesh()->NumNodes());
    constexpr auto tNumFixedNodes = 2U;
    for (std::size_t tIndex = 0; tIndex < tNumFixedNodes; ++tIndex)
    {
        constexpr auto tFixedNodeValue = double{1.0};
        TEST_EQUALITY(tResultOnHost[tIndex], tFixedNodeValue);
    }
    for (std::size_t tIndex = 0; tIndex < tVector.size(); ++tIndex)
    {
        TEST_EQUALITY(tResultOnHost[tIndex + tNumFixedNodes], tVector[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, FullNodalScalarVectorAllBlocks)
{
    const auto tTestMeshFixture = TestMeshSetupTeardown{};

    const auto tMeshDesignVariables = tTestMeshFixture.meshDesignVariablesAllDesignBlocks();
    const auto tResultOnDevice = full_nodal_scalar_vector(tMeshDesignVariables, tTestMeshFixture.mesh());

    TEST_EQUALITY(tVectorDensities.size(), tResultOnDevice.size());

    const auto tResultOnHost =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tResultOnDevice);
    for (auto tIndex = 0U; tIndex < tVectorDensities.size(); ++tIndex)
    {
        TEST_EQUALITY(tResultOnHost[tIndex], tVectorDensities[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, FullNodalScalarVectorBlock2)
{
    const auto tTestMeshFixture = TestMeshSetupTeardown{};

    const auto tMeshDesignVariables = tTestMeshFixture.meshDesignVariablesBlock2Fixed();
    const auto tResultOnDevice = full_nodal_scalar_vector(tMeshDesignVariables, tTestMeshFixture.mesh());
    const auto tResultOnHost =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tResultOnDevice);

    // The last three nodes are in the fixed block so they have density 1.
    // Border nodes belong to the design block.
    auto tExpectedDensities = tVectorDensities;
    tExpectedDensities[5] = 1.0;
    tExpectedDensities[6] = 1.0;
    tExpectedDensities[7] = 1.0;

    for (auto tIndex = 0U; tIndex < tVectorDensities.size(); ++tIndex)
    {
        TEST_EQUALITY(tResultOnHost[tIndex], tExpectedDensities[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, MeshDesignVariablesFromScalarVectorAllBlocks)
{
    const auto tTestMeshFixture = TestMeshSetupTeardown{};

    const auto tBaseMeshDesignVariables = tTestMeshFixture.meshDesignVariablesAllDesignBlocks();
    const auto tControlsOnDevice =
        Plato::ScalarVector{"Controls", static_cast<unsigned>(tTestMeshFixture.mesh()->NumNodes())};
    fill_with_transformed_indices(tControlsOnDevice,
                                  [](const auto tIndex) { return -static_cast<double>(tIndex + 1); });

    const auto tResultMeshDesignVariables =
        mesh_design_variables(tControlsOnDevice, tBaseMeshDesignVariables, tTestMeshFixture.mesh());

    TEST_EQUALITY(tResultMeshDesignVariables.mBlockDensities.size(), tBaseMeshDesignVariables.mBlockDensities.size());

    auto tExpectedDesignVariablesIterator = tBaseMeshDesignVariables.mBlockDensities.cbegin();
    for (const auto& [tResultBlockID, tResultDensityVector] : tResultMeshDesignVariables.mBlockDensities)
    {
        const auto& [tExpectedBlockID, tExpectedDensityVector] = *tExpectedDesignVariablesIterator;
        TEST_EQUALITY(tResultBlockID, tExpectedBlockID);
        TEST_EQUALITY(tResultDensityVector.size(), tExpectedDensityVector.size());

        for (auto tIndex = 0U; tIndex < tResultDensityVector.size(); ++tIndex)
        {
            TEST_EQUALITY(tResultDensityVector[tIndex].mGlobalMeshEntityID,
                          tExpectedDensityVector[tIndex].mGlobalMeshEntityID);
            TEST_EQUALITY(tResultDensityVector[tIndex].mDesignVariableVectorIndex,
                          tExpectedDensityVector[tIndex].mDesignVariableVectorIndex);
            TEST_EQUALITY(tResultDensityVector[tIndex].mDensity, -tExpectedDensityVector[tIndex].mDensity);
        }

        ++tExpectedDesignVariablesIterator;
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, MeshDesignVariablesFromScalarVectorFixedBlock1)
{
    const auto tTestMeshFixture = TestMeshSetupTeardown{};

    const auto tBaseMeshDesignVariables = tTestMeshFixture.meshDesignVariablesBlock1Fixed();
    const auto tControlsOnDevice =
        Plato::ScalarVector{"Controls", static_cast<unsigned>(tTestMeshFixture.mesh()->NumNodes())};
    fill_with_transformed_indices(tControlsOnDevice,
                                  [](const auto tIndex) { return -static_cast<double>(tIndex + 1); });

    const auto tResultMeshDesignVariables =
        mesh_design_variables(tControlsOnDevice, tBaseMeshDesignVariables, tTestMeshFixture.mesh());

    TEST_EQUALITY(tResultMeshDesignVariables.mBlockDensities.size(), tBaseMeshDesignVariables.mBlockDensities.size());

    auto tExpectedDesignVariablesIterator = tBaseMeshDesignVariables.mBlockDensities.cbegin();
    for (const auto& [tResultBlockID, tResultDensityVector] : tResultMeshDesignVariables.mBlockDensities)
    {
        const auto& [tExpectedBlockID, tExpectedDensityVector] = *tExpectedDesignVariablesIterator;
        TEST_EQUALITY(tResultBlockID, tExpectedBlockID);
        TEST_EQUALITY(tResultDensityVector.size(), tExpectedDensityVector.size());

        for (auto tIndex = 0U; tIndex < tResultDensityVector.size(); ++tIndex)
        {
            TEST_EQUALITY(tResultDensityVector[tIndex].mGlobalMeshEntityID,
                          tExpectedDensityVector[tIndex].mGlobalMeshEntityID);
            TEST_EQUALITY(tResultDensityVector[tIndex].mDesignVariableVectorIndex,
                          tExpectedDensityVector[tIndex].mDesignVariableVectorIndex);
            TEST_EQUALITY(tResultDensityVector[tIndex].mDensity, -tExpectedDensityVector[tIndex].mDensity);
        }

        ++tExpectedDesignVariablesIterator;
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, NumberOfDesignVariables)
{
    const auto tTestFixture = TestMeshSetupTeardown{};
    {
        const auto tMeshDesignVariables = plato::mesh::MeshDesignVariables{};
        constexpr auto tSizeForEmptyDesignVariables = 0U;
        TEST_EQUALITY(number_of_design_variables(tMeshDesignVariables), tSizeForEmptyDesignVariables);
    }
    {
        const auto tMeshDesignVariables = tTestFixture.meshDesignVariablesAllDesignBlocks();
        TEST_EQUALITY(number_of_design_variables(tMeshDesignVariables), tTestFixture.mesh()->NumNodes());
    }
    {
        const auto tMeshDesignVariables = tTestFixture.meshDesignVariablesBlock1Fixed();
        TEST_EQUALITY(number_of_design_variables(tMeshDesignVariables), tDensityVector2Block1Fixed.size());
    }
    {
        const auto tMeshDesignVariables = tTestFixture.meshDesignVariablesBlock2Fixed();
        TEST_EQUALITY(number_of_design_variables(tMeshDesignVariables), tDensityVector1Block2Fixed.size());
    }
}

}  // namespace plato::functional::unittest
