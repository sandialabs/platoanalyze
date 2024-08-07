#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <numeric>
#include <plato/filter/FilterInterface.hpp>

#include "BLAS1.hpp"
#include "FunctionalInterfaceUtilities.hpp"
#include "PlatoMeshTestHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTestHelpers.hpp"

namespace plato::functional::unittest
{
namespace
{
const auto tDensityVector1 =
    std::vector<plato::mesh::Density>{{1, 0, 1.0}, {2, 1, 2.0}, {3, 2, 3.0}, {4, 3, 4.0}, {5, 4, 5.0}};
const auto tDensityVector2 =
    std::vector<plato::mesh::Density>{{3, 2, 3.0}, {4, 3, 4.0}, {5, 4, 5.0}, {6, 5, 6.0}, {7, 6, 7.0}, {8, 7, 8.0}};
const auto tVectorDensities = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

Plato::Mesh test_mesh(const std::filesystem::path& aMeshFilePath)
{
    Plato::TestHelpers::write_two_block_mesh(aMeshFilePath);
    return Plato::MeshFactory::create(aMeshFilePath.string());
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

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ToStdVector)
{
    constexpr int tNumEntries = 10;
    const auto tControl = Plato::ScalarVector("test", tNumEntries);
    constexpr double tEntryValue = 42.0;
    Kokkos::deep_copy(tControl, tEntryValue);

    const std::vector tResult = to_std_vector(tControl);
    const auto tExpected = std::vector<double>(tNumEntries, tEntryValue);
    TEST_EQUALITY(tResult.size(), tExpected.size());
    for (std::size_t tIndex = 0; tIndex < tExpected.size(); ++tIndex)
    {
        TEST_EQUALITY(tResult.at(tIndex), tExpected.at(tIndex));
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ToScalarVector)
{
    constexpr int tNumEntries = 10;
    constexpr double tEntryValue = 42.0;
    const auto tVector = std::vector<double>(tNumEntries, tEntryValue);

    const Plato::ScalarVector tResult = to_scalar_vector(tVector);
    const auto tResultOnHost = Kokkos::create_mirror_view(tResult);
    Kokkos::deep_copy(tResultOnHost, tResult);

    TEST_EQUALITY(tResultOnHost.size(), tVector.size());
    for (std::size_t tIndex = 0; tIndex < tVector.size(); ++tIndex)
    {
        TEST_EQUALITY(tResultOnHost[tIndex], tVector[tIndex]);
    }
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

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ToScalarVectorFromMeshDesignVariables)
{
    const auto tMeshFilePath = std::filesystem::path{"test-mesh.exo"};
    const auto tMesh = test_mesh(tMeshFilePath);

    const auto tMeshDesignVariables =
        plato::mesh::MeshDesignVariables{tMeshFilePath, {{1, tDensityVector1}, {2, tDensityVector2}}};

    const auto tResultOnDevice = to_scalar_vector(tMeshDesignVariables, tMesh);

    TEST_EQUALITY(tVectorDensities.size(), tResultOnDevice.size());

    const auto tResultOnHost = Kokkos::create_mirror_view(tResultOnDevice);
    Kokkos::deep_copy(tResultOnHost, tResultOnDevice);
    for (auto tIndex = 0U; tIndex < tVectorDensities.size(); ++tIndex)
    {
        TEST_EQUALITY(tResultOnHost[tIndex], tVectorDensities[tIndex]);
    }

    std::filesystem::remove(tMeshFilePath);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ToScalarVectorFromMeshDesignVariablesFixedBlock)
{
    const auto tMeshFilePath = std::filesystem::path{"test-mesh.exo"};
    const auto tMesh = test_mesh(tMeshFilePath);

    // Omits block 1, which is then assumed fixed.
    const auto tMeshDesignVariables = plato::mesh::MeshDesignVariables{tMeshFilePath, {{1, tDensityVector1}}};

    const auto tResultOnDevice = to_scalar_vector(tMeshDesignVariables, tMesh);
    const auto tResultOnHost = Kokkos::create_mirror_view(tResultOnDevice);
    Kokkos::deep_copy(tResultOnHost, tResultOnDevice);

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

    std::filesystem::remove(tMeshFilePath);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ToMeshDesignVariablesFromScalarVector)
{
    const auto tMeshFilePath = std::filesystem::path{"test-mesh.exo"};
    const auto tMesh = test_mesh(tMeshFilePath);

    const auto tBaseMeshDesignVariables =
        plato::mesh::MeshDesignVariables{tMeshFilePath, {{1, tDensityVector1}, {2, tDensityVector2}}};

    const auto tControlsOnDevice = Plato::ScalarVector{"Controls", static_cast<unsigned>(tMesh->NumNodes())};
    const auto tControlsOnHost = Kokkos::create_mirror_view(tControlsOnDevice);
    for (auto tIndex = 0U; tIndex < tControlsOnHost.size(); ++tIndex)
    {
        tControlsOnHost[tIndex] = -static_cast<double>(tIndex + 1);
    }
    Kokkos::deep_copy(tControlsOnDevice, tControlsOnHost);

    const auto tResultMeshDesignVariables = to_mesh_design_variables(tControlsOnDevice, tBaseMeshDesignVariables);

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

    std::filesystem::remove(tMeshFilePath);
}

}  // namespace plato::functional::unittest
