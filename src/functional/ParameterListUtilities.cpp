#include "functional/ParameterListUtilities.hpp"

namespace plato::functional
{
auto element_block_names(const Teuchos::ParameterList& aParameterList) -> std::vector<std::string>
{
    auto tBlockNames = std::vector<std::string>{};
    const auto& tDomainSublist = all_domains_sublist(aParameterList);
    tBlockNames.reserve(tDomainSublist.numParams());
    std::transform(
        tDomainSublist.begin(), tDomainSublist.end(), std::back_inserter(tBlockNames),
        [](const auto& aEntry) {
            return Teuchos::getValue<Teuchos::ParameterList>(aEntry.second).template get<std::string>("Element Block");
        });
    return tBlockNames;
}
}  // namespace plato::functional
