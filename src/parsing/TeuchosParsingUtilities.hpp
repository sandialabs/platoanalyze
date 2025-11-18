#ifndef PLATO_UTILITIES_TEUCHOSPARSINGUTILITIES_H
#define PLATO_UTILITIES_TEUCHOSPARSINGUTILITIES_H

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_TestForException.hpp>
#include <stdexcept>
#include <string>

namespace plato::utilities
{
/// @brief loops through entries in Teuchos list @param ParentList checks that each entry is itself a list and applies
/// @param aSublistFunc to each sublist.
template <typename SublistFunc>
void for_each_sublist(const Teuchos::ParameterList& aParentList, const SublistFunc& aSublistFunc)
{
    for (auto& tSublistKeyValue : aParentList)
    {
        const Teuchos::ParameterEntry& tSublist = tSublistKeyValue.second;
        TEUCHOS_TEST_FOR_EXCEPTION(!tSublist.isList(), std::logic_error,
                                   std::string{"Invalid parameter in ParameterList '"} + aParentList.name() +
                                       std::string{"'. Expected only list parameters."});
        const std::string tName = tSublistKeyValue.first;

        aSublistFunc(tName);
    }
}
}  // namespace plato::utilities
#endif
