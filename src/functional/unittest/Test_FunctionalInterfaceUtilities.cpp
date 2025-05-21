#include <Kokkos_StdAlgorithms.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <numeric>
#include <plato/filter/library/FilterInterface.hpp>
#include <random>

#include "BLAS1.hpp"
#include "FunctionalInterfaceUtilities.hpp"
#include "PlatoMeshTestHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTestHelpers.hpp"
#include "TestMeshSetupTeardown.hpp"

namespace plato::functional::unittest
{
namespace
{
const auto tVectorDensities = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

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

auto iota_scalar_vector(const std::size_t aSize) -> Plato::ScalarVector
{
    const auto tHostVec = Plato::HostScalarVector{"host_vec", aSize};
    std::iota(Kokkos::Experimental::begin(tHostVec), Kokkos::Experimental::end(tHostVec), 0.0);
    const auto tDeviceVec = Plato::ScalarVector{"device_vec", aSize};
    Kokkos::deep_copy(tDeviceVec, tHostVec);
    return tDeviceVec;
}

template <typename Value>
void test_equality_vectors(const std::vector<Value>& aVector1,
                           const std::vector<Value>& aVector2,
                           Teuchos::FancyOStream& aOutStream,
                           bool& aSuccess)
{
    TEUCHOS_TEST_EQUALITY(aVector1.size(), aVector2.size(), aOutStream, aSuccess);
    for (size_t i = 0; i < aVector1.size(); ++i)
    {
        TEUCHOS_TEST_EQUALITY(aVector1[i], aVector2[i], aOutStream, aSuccess);
    }
}

}  // namespace

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ParameterList)
{
    constexpr double tPhysicalFilterRadius = 42.0;
    const double tHelmholtzFilterRadius = helmholtz_radius_from_physical_radius(tPhysicalFilterRadius);
    constexpr double tBoundaryStickingPenalty = 13.0;
    const auto tFilterParameters =
        filter::library::FilterParameters{/*.mFilterRadius=*/tPhysicalFilterRadius,
                                          /*.mBoundaryStickingPenalty=*/tBoundaryStickingPenalty};
    constexpr auto tMeshName = std::string_view{"not-a-mesh.exo"};
    const Teuchos::ParameterList tParameterList = helmholtz_filter_parameter_list(tFilterParameters, tMeshName, {});
    TEST_EQUALITY(tParameterList.get<std::string>("Physics"), "Plato Driver");
    TEST_EQUALITY(tParameterList.sublist("Plato Problem").get<std::string>("Physics"), "Helmholtz Filter");
    TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters").get<double>("Length Scale"),
                  tHelmholtzFilterRadius);
    TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters").get<double>("Surface Length Scale"),
                  tBoundaryStickingPenalty);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AdjointHelmholtzParameterList)
{
    const auto tFilterParameters = filter::library::FilterParameters{};
    constexpr auto tMeshName = std::string_view{"not-a-mesh.exo"};
    const auto tAdjointParameterList = adjoint_helmholtz_filter_parameter_list(tFilterParameters, tMeshName, {});

    TEST_EQUALITY(tAdjointParameterList.sublist("Plato Problem").get<std::string>("Physics"),
                  "Adjoint Helmholtz Filter");

    const auto tParameterList = helmholtz_filter_parameter_list(tFilterParameters, tMeshName, {});
    TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters"),
                  tAdjointParameterList.sublist("Plato Problem").sublist("Parameters"));
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, UpdateMesh)
{
    constexpr std::string_view tInitialMeshName = "first-mesh-name.exo";
    Teuchos::ParameterList tParameterList =
        helmholtz_filter_parameter_list(filter::library::FilterParameters{}, tInitialMeshName, {});

    TEST_EQUALITY(tParameterList.get<std::string>("Input Mesh"), std::string{tInitialMeshName});

    constexpr std::string_view tNewMeshName = "second-mesh-name.exo";
    update_mesh_file_name(tParameterList, tNewMeshName);
    TEST_EQUALITY(tParameterList.get<std::string>("Input Mesh"), std::string{tNewMeshName});
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, HelmholtzFilterParameterList)
{
    const auto tBlock1Name = std::string{"block_1"};
    const auto tBlock42Name = std::string{"the_answer"};
    constexpr auto tMeshName = std::string_view{"not-a-mesh.exo"};
    const auto tParameterList =
        helmholtz_filter_parameter_list(filter::library::FilterParameters{}, tMeshName, {tBlock1Name, tBlock42Name});

    TEST_EQUALITY(tParameterList.get<std::string>("Physics"), "Plato Driver");

    const auto& tDomains = tParameterList.sublist("Plato Problem").sublist("Spatial Model").sublist("Domains");
    TEST_ASSERT(tDomains.isSublist(tBlock1Name));
    TEST_ASSERT(tDomains.isSublist(tBlock42Name));
    TEST_ASSERT(!tDomains.isSublist("block_2"));  // Arbitrary, but not unlikely

    const auto& tBlock1Sublist = tDomains.sublist(tBlock1Name);
    const auto tElementBlockSublist = std::string{"Element Block"};
    TEST_EQUALITY(tBlock1Sublist.get<std::string>(tElementBlockSublist), tBlock1Name);

    const auto& tBlock42Sublist = tDomains.sublist(tBlock42Name);
    TEST_EQUALITY(tBlock42Sublist.get<std::string>(tElementBlockSublist), tBlock42Name);

    const auto tMaterialModelSublist = std::string{"Material Model"};
    const auto tExpectedMaterialModelName = std::string{"material_1"};
    TEST_EQUALITY(tBlock1Sublist.get<std::string>(tMaterialModelSublist), tExpectedMaterialModelName);
    TEST_EQUALITY(tBlock42Sublist.get<std::string>(tMaterialModelSublist), tExpectedMaterialModelName);
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

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, DesignVariableStdVector)
{
    const auto tTestFixture = TestMeshSetupTeardown{};

    const auto tTestFunction =
        [&](const std::vector<double>& aResult, const auto aFullControls, const auto aNumberOfFixedNodes)
    {
        const auto tExpectedSize = aFullControls.size() - aNumberOfFixedNodes;
        const auto tExpected = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, aFullControls);
        TEST_EQUALITY(aResult.size(), tExpectedSize);
        for (std::size_t tIndex = aNumberOfFixedNodes; tIndex < aFullControls.size(); ++tIndex)
        {
            TEST_EQUALITY(aResult.at(tIndex - aNumberOfFixedNodes), tExpected(tIndex));
        }
    };

    const auto tNumEntries = tTestFixture.mesh()->NumNodes();
    const auto tControl = Plato::ScalarVector("test", tNumEntries);
    fill_with_transformed_indices(tControl, [](const auto tIndex) { return static_cast<double>(tIndex); });
    // All blocks
    {
        const auto tResult =
            design_variable_std_vector(tControl, tTestFixture.analysisDomainMeshAllDesignBlocks(), tTestFixture.mesh());

        constexpr auto tNumberOfFixedNodes = 0U;
        tTestFunction(tResult, tControl, tNumberOfFixedNodes);
    }
    // Fixed block
    {
        const auto tResult =
            design_variable_std_vector(tControl, tTestFixture.analysisDomainMeshBlock1Fixed(), tTestFixture.mesh());

        constexpr auto tNumberOfFixedNodes = 2U;
        tTestFunction(tResult, tControl, tNumberOfFixedNodes);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, FullNodalScalarVectorFromVectorAllBlocks)
{
    const auto tTestFixture = TestMeshSetupTeardown{};
    const auto tVector = std::vector<double>(tTestFixture.mesh()->NumNodes(), 42.0);

    const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshAllDesignBlocks();

    constexpr auto tFillValue = 1.0;
    const auto tResult = full_nodal_scalar_vector(tVector, tAnalysisDomainMesh, tTestFixture.mesh(), tFillValue);
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

    const auto tVector = tTestFixture.densityValuesBlock1Fixed();
    const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock1Fixed();

    constexpr auto tFixedNodeValue = double{1.0};
    const auto tResult = full_nodal_scalar_vector(tVector, tAnalysisDomainMesh, tTestFixture.mesh(), tFixedNodeValue);
    const auto tResultOnHost = Kokkos::create_mirror_view(tResult);
    Kokkos::deep_copy(tResultOnHost, tResult);

    TEST_EQUALITY(tResultOnHost.size(), tTestFixture.mesh()->NumNodes());
    constexpr auto tNumFixedNodes = 2U;
    for (std::size_t tIndex = 0; tIndex < tNumFixedNodes; ++tIndex)
    {
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

    const auto tAnalysisDomainMesh = tTestMeshFixture.analysisDomainMeshAllDesignBlocks();
    const auto tResultOnDevice = full_nodal_scalar_vector(tAnalysisDomainMesh, tTestMeshFixture.mesh());

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

    const auto tAnalysisDomainMesh = tTestMeshFixture.analysisDomainMeshBlock2Fixed();
    const auto tResultOnDevice = full_nodal_scalar_vector(tAnalysisDomainMesh, tTestMeshFixture.mesh());
    const auto tResultOnHost =
        Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace{}, tResultOnDevice);

    // The last three nodes are in the fixed block so they have density tFillValue.
    // Border nodes belong to the design block.
    constexpr auto tFillValue = double{1.0};
    auto tExpectedDensities = tVectorDensities;
    tExpectedDensities[5] = tFillValue;
    tExpectedDensities[6] = tFillValue;
    tExpectedDensities[7] = tFillValue;

    for (auto tIndex = 0U; tIndex < tVectorDensities.size(); ++tIndex)
    {
        TEST_EQUALITY(tResultOnHost[tIndex], tExpectedDensities[tIndex]);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, AnalysisDomainMeshFromScalarVector)
{
    const auto tTestMeshFixture = TestMeshSetupTeardown{};

    const auto tTestFunction = [&](const plato::analysis::AnalysisDomainMesh& aResultAnalysisDomainMesh,
                                   const plato::analysis::AnalysisDomainMesh& aBaseAnalysisDomainMesh)
    {
        TEST_EQUALITY(aResultAnalysisDomainMesh.mBlockScalarField.size(),
                      aBaseAnalysisDomainMesh.mBlockScalarField.size());

        auto tExpectedDesignVariablesIterator = aBaseAnalysisDomainMesh.mBlockScalarField.cbegin();
        for (const auto& [tResultBlockID, tResultDensityVector] : aResultAnalysisDomainMesh.mBlockScalarField)
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
                TEST_EQUALITY(tResultDensityVector[tIndex].mValue, -tExpectedDensityVector[tIndex].mValue);
            }

            ++tExpectedDesignVariablesIterator;
        }
    };

    const auto tControlsOnDevice =
        Plato::ScalarVector{"Controls", static_cast<unsigned>(tTestMeshFixture.mesh()->NumNodes())};
    fill_with_transformed_indices(tControlsOnDevice,
                                  [](const auto tIndex) { return -static_cast<double>(tIndex + 1); });
    // All blocks
    {
        const auto tBaseAnalysisDomainMesh = tTestMeshFixture.analysisDomainMeshAllDesignBlocks();
        const auto tResultAnalysisDomainMesh =
            analysis_domain_mesh(tControlsOnDevice, tBaseAnalysisDomainMesh, tTestMeshFixture.mesh());
        tTestFunction(tResultAnalysisDomainMesh, tBaseAnalysisDomainMesh);
    }
    // Block 1 fixed
    {
        const auto tBaseAnalysisDomainMesh = tTestMeshFixture.analysisDomainMeshBlock1Fixed();
        const auto tResultAnalysisDomainMesh =
            analysis_domain_mesh(tControlsOnDevice, tBaseAnalysisDomainMesh, tTestMeshFixture.mesh());
        tTestFunction(tResultAnalysisDomainMesh, tBaseAnalysisDomainMesh);
    }
    // Block 2 fixed
    {
        const auto tBaseAnalysisDomainMesh = tTestMeshFixture.analysisDomainMeshBlock2Fixed();
        const auto tResultAnalysisDomainMesh =
            analysis_domain_mesh(tControlsOnDevice, tBaseAnalysisDomainMesh, tTestMeshFixture.mesh());
        tTestFunction(tResultAnalysisDomainMesh, tBaseAnalysisDomainMesh);
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, NumberOfDesignVariables)
{
    const auto tTestFixture = TestMeshSetupTeardown{};
    {
        const auto tAnalysisDomainMesh = plato::analysis::AnalysisDomainMesh{};
        constexpr auto tSizeForEmptyDesignVariables = 0U;
        TEST_EQUALITY(number_of_analysis_field_variables(tAnalysisDomainMesh), tSizeForEmptyDesignVariables);
    }
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshAllDesignBlocks();
        TEST_EQUALITY(number_of_analysis_field_variables(tAnalysisDomainMesh), tTestFixture.mesh()->NumNodes());
    }
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock1Fixed();
        TEST_EQUALITY(number_of_analysis_field_variables(tAnalysisDomainMesh),
                      tTestFixture.numberOfAnalysisDomainMeshBlock1Fixed());
    }
    {
        const auto tAnalysisDomainMesh = tTestFixture.analysisDomainMeshBlock2Fixed();
        TEST_EQUALITY(number_of_analysis_field_variables(tAnalysisDomainMesh),
                      tTestFixture.numberOfAnalysisDomainMeshBlock2Fixed());
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ScalarVectorToStdVectorTrivialNodeMap)
{
    const auto tNodeMap =
        std::unordered_map<Plato::OrdinalType, Plato::OrdinalType>{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}};

    constexpr auto tDimension = 1U;
    const auto tCopy =
        scalar_vector_to_std_vector_sorted_by_global_id(iota_scalar_vector(tNodeMap.size()), tNodeMap, tDimension);

    const auto tExpected = std::vector{0.0, 1.0, 2.0, 3.0, 4.0};
    test_equality_vectors(tCopy, tExpected, out, success);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ScalarVectorToStdVectorNonTrivialNodeMap)
{
    const auto tNodeMap = std::unordered_map<Plato::OrdinalType, Plato::OrdinalType>{{10, 0}, {11, 1}, {2, 3}, {5, 2}};

    constexpr auto tDimension = 1U;
    const auto tResult =
        scalar_vector_to_std_vector_sorted_by_global_id(iota_scalar_vector(tNodeMap.size()), tNodeMap, tDimension);

    const auto tExpected = std::vector{3.0, 2.0, 0.0, 1.0};
    test_equality_vectors(tResult, tExpected, out, success);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ScalarVectorToStdVectorNonTrivialNodeMap3D)
{
    const auto tNodeMap =
        std::unordered_map<Plato::OrdinalType, Plato::OrdinalType>{{7, 0}, {1, 1}, {8, 2}, {9, 3}, {3, 4}};
    constexpr auto tDimension = std::size_t{3U};
    const auto tTotalSize = tNodeMap.size() * tDimension;
    const auto tDeviceVec = iota_scalar_vector(tTotalSize);

    const auto tResult = scalar_vector_to_std_vector_sorted_by_global_id(tDeviceVec, tNodeMap, tDimension);

    const auto tExpected = std::vector{3.0, 4.0, 5.0, 12.0, 13.0, 14.0, 0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0};
    test_equality_vectors(tResult, tExpected, out, success);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, SortedMapVector)
{
    const auto tExpected = std::vector<std::pair<Plato::OrdinalType, Plato::OrdinalType>>{
        {1, 1}, {3, 4}, {7, 0}, {8, 2}, {9, 3},
    };
    auto tMap = std::unordered_map<Plato::OrdinalType, Plato::OrdinalType>{};
    std::copy(tExpected.begin(), tExpected.end(), std::inserter(tMap, tMap.begin()));

    const auto tSortedMapVector = detail::sorted_map_vector(tMap);

    test_equality_vectors(tSortedMapVector, tExpected, out, success);
}

}  // namespace plato::functional::unittest
