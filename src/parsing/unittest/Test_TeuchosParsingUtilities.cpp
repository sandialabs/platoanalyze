#include <mpi.h>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "parsing/TeuchosParsingUtilities.hpp"

namespace plato::utilities::unittest
{
TEUCHOS_UNIT_TEST(TeuchosParsingUtilities, ForEachSublist_ThrowsIfNotSublist)
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Criteria");
    tParameterList.sublist("Strain Energy");
    tParameterList.set("Thermal Energy", "potato");

    auto tDoNothing = [](const std::string& tName) {};
    TEST_THROW(for_each_sublist(tParameterList, tDoNothing), std::logic_error);
}

TEUCHOS_UNIT_TEST(TeuchosParsingUtilities, ForEachSublist_CorrectSublistNames)
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Criteria");
    tParameterList.sublist("Strain Energy");
    tParameterList.sublist("Thermal Energy");

    const std::vector<std::string> tGoldNames{"Strain Energy", "Thermal Energy"};
    std::vector<std::string> tOutNames{};

    auto tCheckNames = [&tOutNames](const std::string& tName) { tOutNames.push_back(tName); };
    for_each_sublist(tParameterList, tCheckNames);

    TEST_ASSERT(tGoldNames.size() == tOutNames.size());
    for (std::size_t tOrdinal = 0; tOrdinal < tGoldNames.size(); tOrdinal++)
    {
        TEST_EQUALITY(tGoldNames[tOrdinal], tOutNames[tOrdinal]);
    }
}
}  // namespace plato::utilities::unittest
