#include <Teuchos_UnitTestHarness.hpp>
#include <plato/filter/FilterInterface.hpp>

#include "ParameterListUtilities.hpp"

namespace plato::functional::unittest
{
namespace
{
const auto kPlatoProblemLabel = std::string{"Plato Problem"};
const auto kSpatialDomainLabel = std::string{"Spatial Model"};
const auto kDomainsLabel = std::string{"Domains"};
const auto kElementBlockLabel = std::string{"Element Block"};
const auto kParametersLabel = std::string{"Parameters"};

/// @brief Adds the sublist path given by @a aSublistPathToTest to an empty ParameterList and checks that @a
/// aSublistFunction returns the sublist corresponding to that path.
template <typename SublistFunction>
auto test_sublist_path_with_const_parameter_list(const SublistFunction& aSublistFunction,
                                                 const std::vector<std::string>& aSublistPathToTest) -> bool
{
    auto tParameterList = Teuchos::ParameterList{};
    auto tSublist = std::ref(tParameterList);
    for (const auto& tSublistLabel : aSublistPathToTest)
    {
        tSublist = tSublist.get().sublist(tSublistLabel);
    }
    const auto tAdditionalSublistName = std::string{"bananas"};
    tSublist.get().sublist(tAdditionalSublistName);

    const auto tConstParameterList = tParameterList;
    const auto& tSublistFromFunction = aSublistFunction(tConstParameterList);
    return tSublistFromFunction.isSublist(tAdditionalSublistName);
}

/// @brief Passes an empty ParameterList to @a aSublistFunction, and checks that the expected path give by @a
/// aSublistPath is created.
template <typename SublistFunction>
auto test_sublist_path_with_nonconst_parameter_list(const SublistFunction& aSublistFunction,
                                                    std::vector<std::string> aSublistPathToTest,
                                                    Teuchos::FancyOStream& aOut,
                                                    bool& aSuccess)
{
    auto tParameterList = Teuchos::ParameterList{};
    auto& tSublistFromFunction = aSublistFunction(tParameterList);

    const auto tAdditionalSublistName = std::string{"apples"};
    tSublistFromFunction.sublist(tAdditionalSublistName);
    aSublistPathToTest.push_back(tAdditionalSublistName);

    // Test that the original has the expected path and the new sublist added to the returned sublist
    auto tSublist = std::ref(tParameterList);
    for (const auto& tSublistLabel : aSublistPathToTest)
    {
        TEUCHOS_TEST_ASSERT(tSublist.get().isSublist(tSublistLabel), aOut, aSuccess);
        tSublist = tSublist.get().sublist(tSublistLabel);
    }
}

}  // namespace

TEUCHOS_UNIT_TEST(ParameterListUtilities, PlatoProblemSublist)
{
    // const ParameterList
    TEST_ASSERT(test_sublist_path_with_const_parameter_list(
        [](const auto& tParameterList) { return plato_problem_sublist(tParameterList); }, {kPlatoProblemLabel}));

    // non-const ParameterList
    test_sublist_path_with_nonconst_parameter_list([](auto& tParameterList) -> Teuchos::ParameterList&
                                                   { return plato_problem_sublist(tParameterList); },
                                                   {kPlatoProblemLabel}, out, success);
}

TEUCHOS_UNIT_TEST(ParameterListUtilities, AllDomainsSublist)
{
    const auto tSublistPath = std::vector{kPlatoProblemLabel, kSpatialDomainLabel, kDomainsLabel};
    // Const
    TEST_ASSERT(test_sublist_path_with_const_parameter_list(
        [](const auto& tParameterList) { return all_domains_sublist(tParameterList); }, tSublistPath));

    // Non-const
    test_sublist_path_with_nonconst_parameter_list([](auto& tParameterList) -> Teuchos::ParameterList&
                                                   { return all_domains_sublist(tParameterList); },
                                                   tSublistPath, out, success);
}

TEUCHOS_UNIT_TEST(ParameterListUtilities, DomainSublist)
{
    const auto tTestDomainName = std::string{"blueberries"};
    const auto tSublistPath = std::vector{kPlatoProblemLabel, kSpatialDomainLabel, kDomainsLabel, tTestDomainName};
    // Const
    TEST_ASSERT(test_sublist_path_with_const_parameter_list([&tTestDomainName](const auto& tParameterList)
                                                            { return domain_sublist(tParameterList, tTestDomainName); },
                                                            tSublistPath));

    // Non-const
    test_sublist_path_with_nonconst_parameter_list([&tTestDomainName](auto& tParameterList) -> Teuchos::ParameterList&
                                                   { return domain_sublist(tParameterList, tTestDomainName); },
                                                   tSublistPath, out, success);
}

TEUCHOS_UNIT_TEST(ParameterListUtilities, ParametersSublist)
{
    const auto tSublistPath = std::vector{kPlatoProblemLabel, kParametersLabel};
    // Const
    TEST_ASSERT(test_sublist_path_with_const_parameter_list(
        [](const auto& tParameterList) { return parameters_sublist(tParameterList); }, tSublistPath));

    // Non-const
    test_sublist_path_with_nonconst_parameter_list([](auto& tParameterList) -> Teuchos::ParameterList&
                                                   { return parameters_sublist(tParameterList); },
                                                   tSublistPath, out, success);
}

TEUCHOS_UNIT_TEST(ParameterListUtilities, BlockNames)
{
    auto tParameterList = Teuchos::ParameterList{};
    const auto tDomain1BlockName = std::string{"Eagle"};
    tParameterList.sublist(kPlatoProblemLabel)
        .sublist(kSpatialDomainLabel)
        .sublist(kDomainsLabel)
        .sublist("Domain 1")
        .set(kElementBlockLabel, tDomain1BlockName);
    const auto tDomain2BlockName = std::string{"Falcon"};
    tParameterList.sublist(kPlatoProblemLabel)
        .sublist(kSpatialDomainLabel)
        .sublist(kDomainsLabel)
        .sublist("Domain 2")
        .set(kElementBlockLabel, tDomain2BlockName);

    const auto tBlockNames = element_block_names(tParameterList);
    constexpr auto tExpectedNumberOfNames = 2U;

    TEST_EQUALITY(tBlockNames.size(), tExpectedNumberOfNames);
    TEST_EQUALITY_CONST(tBlockNames.front(), tDomain1BlockName);
    TEST_EQUALITY_CONST(tBlockNames.back(), tDomain2BlockName);
}
}  // namespace plato::functional::unittest
