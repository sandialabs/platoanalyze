#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <numeric>

#include "BLAS1.hpp"
#include "FilterInterface.hpp"
#include "FunctionalInterfaceUtilities.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTestHelpers.hpp"

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ParameterList)
{
    namespace pf = Plato::Functional;
    constexpr double tFilterRadius = 42.0;
    constexpr double tBoundaryStickingPenalty = 13.0;
    const auto tFilterParameters =
        plato::functional::filter::library::FilterParameters{/*.mFilterRadius=*/tFilterRadius,
                                                             /*.mBoundaryStickingPenalty=*/tBoundaryStickingPenalty};
    constexpr auto tMeshName = std::string_view{"not-a-mesh.exo"};
    const Teuchos::ParameterList tParameterList =
        plato::functional::helmholtz_filter_parameter_list(tFilterParameters, tMeshName);

    TEST_EQUALITY(tParameterList.get<std::string>("Physics"), "Plato Driver");
    TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters").get<double>("Length Scale"),
                  tFilterRadius);
    TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters").get<double>("Surface Length Scale"),
                  tBoundaryStickingPenalty);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, UpdateMesh)
{
    namespace pf = Plato::Functional;
    constexpr std::string_view tInitialMeshName = "first-mesh-name.exo";
    Teuchos::ParameterList tParameterList = plato::functional::helmholtz_filter_parameter_list(
        plato::functional::filter::library::FilterParameters{}, tInitialMeshName);

    TEST_EQUALITY(tParameterList.get<std::string>("Input Mesh"), std::string{tInitialMeshName});

    constexpr std::string_view tNewMeshName = "second-mesh-name.exo";
    plato::functional::update_mesh_file_name(tParameterList, tNewMeshName);
    TEST_EQUALITY(tParameterList.get<std::string>("Input Mesh"), std::string{tNewMeshName});
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ToStdVector)
{
    namespace pf = Plato::Functional;
    constexpr int tNumEntries = 10;
    const auto tControl = Plato::ScalarVector("test", tNumEntries);
    constexpr double tEntryValue = 42.0;
    Kokkos::deep_copy(tControl, tEntryValue);

    const std::vector tResult = plato::functional::to_std_vector(tControl);
    const auto tExpected = std::vector<double>(tNumEntries, tEntryValue);
    TEST_EQUALITY(tResult.size(), tExpected.size());
    for (std::size_t tIndex = 0; tIndex < tExpected.size(); ++tIndex)
    {
        TEST_EQUALITY(tResult.at(tIndex), tExpected.at(tIndex));
    }
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ToScalarVector)
{
    namespace pf = Plato::Functional;
    constexpr int tNumEntries = 10;
    constexpr double tEntryValue = 42.0;
    const auto tVector = std::vector<double>(tNumEntries, tEntryValue);

    const Plato::ScalarVector tResult = plato::functional::to_scalar_vector(tVector);
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
    namespace pf = Plato::Functional;

    constexpr int tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    constexpr unsigned int tSize = 10;
    const auto tControl = Plato::ScalarVector("test", tSize);
    Plato::blas1::fill(1.0, tControl);
    auto tOriginalHash = plato::functional::hash_current_design(tControl, tMesh);

    auto tNewHash = plato::functional::hash_current_design(tControl, tMesh);
    TEST_EQUALITY(tNewHash, tOriginalHash);

    constexpr int tNewMeshWidth = 2;
    tMesh = Plato::TestHelpers::get_box_mesh("TET4", tNewMeshWidth);
    tNewHash = plato::functional::hash_current_design(tControl, tMesh);
    TEST_INEQUALITY(tNewHash, tOriginalHash);

    // regenerate the original mesh and ensure that the hash matches original
    tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    tNewHash = plato::functional::hash_current_design(tControl, tMesh);
    TEST_EQUALITY(tNewHash, tOriginalHash);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, HashCurrentDesign_ControlChanges)
{
    namespace pf = Plato::Functional;

    constexpr int tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    constexpr unsigned int tSize = 10;
    const auto tControl = Plato::ScalarVector("test", tSize);
    Plato::blas1::fill(1.0, tControl);
    auto tOriginalHash = plato::functional::hash_current_design(tControl, tMesh);

    auto tNewHash = plato::functional::hash_current_design(tControl, tMesh);
    TEST_EQUALITY(tNewHash, tOriginalHash);

    Plato::blas1::fill(0.5, tControl);
    tNewHash = plato::functional::hash_current_design(tControl, tMesh);
    TEST_INEQUALITY(tNewHash, tOriginalHash);
}
