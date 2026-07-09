#include "problem/geometric/test_utilities/MassPropertiesCriterionUtilities.hpp"

#include <string_view>

namespace plato::problem::geometric::test_utilities
{
namespace
{

constexpr auto kMassPropertiesName = std::string_view{"Mass Properties"};
}

auto mass_properties_criterion(const Teuchos::Array<std::string>& aPropertyList,
                               const Teuchos::Array<double>& aWeightsList,
                               const Teuchos::Array<double>& aGoldValuesList,
                               const unsigned int aPower) -> Teuchos::ParameterList
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Criteria");
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Type", std::string{kMassPropertiesName});
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Properties", aPropertyList);
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Weights", aWeightsList);
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Gold Values", aGoldValuesList);
    tParameterList.sublist(std::string{kMassPropertiesName}).set("Least Squares Exponent", aPower);
    return tParameterList;
}
}  // namespace plato::problem::geometric::test_utilities
