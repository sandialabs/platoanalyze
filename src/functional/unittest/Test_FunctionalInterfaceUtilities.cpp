#include "FunctionalInterfaceUtilities.hpp"
#include "FilterInterface.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <numeric>

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ParameterList)
{
  namespace pf = Plato::Functional;
  constexpr double tFilterRadius = 42.0;
  constexpr double tBoundaryStickingPenalty = 13.0;
  const auto tFilterParameters = pf::FilterParameters{
    /*.mFilterRadius=*/tFilterRadius,
    /*.mBoundaryStickingPenalty=*/tBoundaryStickingPenalty};
  constexpr auto tMeshName = std::string_view{"not-a-mesh.exo"};
  const Teuchos::ParameterList tParameterList = 
    pf::helmholtz_filter_parameter_list(tFilterParameters, tMeshName);

  TEST_EQUALITY(tParameterList.get<std::string>("Physics"), "Plato Driver");
  TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters").get<double>("Length Scale"), tFilterRadius);
  TEST_EQUALITY(tParameterList.sublist("Plato Problem").sublist("Parameters").get<double>("Surface Length Scale"), tBoundaryStickingPenalty);
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, UpdateMesh)
{
  namespace pf = Plato::Functional;
  constexpr std::string_view tInitialMeshName = "first-mesh-name.exo";
  Teuchos::ParameterList tParameterList = 
    pf::helmholtz_filter_parameter_list(pf::FilterParameters{}, tInitialMeshName);

  TEST_EQUALITY(tParameterList.get<std::string>("Input Mesh"), std::string{tInitialMeshName});
  
  constexpr std::string_view tNewMeshName = "second-mesh-name.exo";
  pf::update_mesh_file_name(tParameterList, tNewMeshName);
  TEST_EQUALITY(tParameterList.get<std::string>("Input Mesh"), std::string{tNewMeshName});
}

TEUCHOS_UNIT_TEST(FunctionalInterfaceUtilities, ToStdVector)
{
  namespace pf = Plato::Functional;
  constexpr int tNumEntries = 10;
  const auto tControl = Plato::ScalarVector("test", tNumEntries);
  constexpr double tEntryValue = 42.0;
  Kokkos::deep_copy(tControl, tEntryValue);

  const std::vector tResult = pf::to_std_vector(tControl);
  const auto tExpected = std::vector<double>(tNumEntries, tEntryValue);
  TEST_EQUALITY(tResult.size(), tExpected.size());
  for(std::size_t tIndex = 0; tIndex < tExpected.size(); ++tIndex)
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

  const Plato::ScalarVector tResult = pf::to_scalar_vector(tVector);
  const auto tResultOnHost = Kokkos::create_mirror_view(tResult);
  Kokkos::deep_copy(tResultOnHost, tResult);

  TEST_EQUALITY(tResultOnHost.size(), tVector.size());
  for(std::size_t tIndex = 0; tIndex < tVector.size(); ++tIndex)
  {
    TEST_EQUALITY(tResultOnHost[tIndex], tVector[tIndex]);
  }
}
