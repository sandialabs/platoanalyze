#ifndef PLATO_FUNCTIONAL_PARAMETERLISTUTILITIES_H
#define PLATO_FUNCTIONAL_PARAMETERLISTUTILITIES_H

#include <Teuchos_ParameterList.hpp>
#include <string>

namespace plato::functional
{
/// @brief Returns the sublist associated with `Plato Problem`.
/// @note If @a ParameterListType is a non-const l-value reference, then this will modify @a aParameterList by adding
/// the sublist if it does not exist.
template <typename ParameterListType>
decltype(auto) plato_problem_sublist(ParameterListType&& aParameterList);

/// @brief Returns the sublist associated with `Plato Problem::Spatial Model::Domains`.
/// @note If @a ParameterListType is a non-const l-value reference, then this will modify @a aParameterList by adding
/// the sublist if it does not exist.
template <typename ParameterListType>
decltype(auto) all_domains_sublist(ParameterListType&& aParameterList);

/// @brief Returns the sublist associated with `Plato Problem::Spatial Model::Domains` appendend with @a aDomainName.
/// @note If @a ParameterListType is a non-const l-value reference, then this will modify @a aParameterList by adding
/// the sublist if it does not exist.
template <typename ParameterListType>
decltype(auto) domain_sublist(ParameterListType&& aParameterList, const std::string_view aDomainName);

/// @brief Returns the sublist associated with `Plato Problem::Parameters`.
/// @note If @a ParameterListType is a non-const l-value reference, then this will modify @a aParameterList by adding
/// the sublist if it does not exist.
template <typename ParameterListType>
decltype(auto) parameters_sublist(ParameterListType&& aParameterList);

/// @brief Returns the names of all element blocks in `Plato Problem::Spatial Model::Domains`.
auto element_block_names(const Teuchos::ParameterList& aParameterList) -> std::vector<std::string>;

template <typename ParameterListType>
decltype(auto) plato_problem_sublist(ParameterListType&& aParameterList)
{
    return aParameterList.sublist("Plato Problem");
}

template <typename ParameterListType>
decltype(auto) all_domains_sublist(ParameterListType&& aParameterList)
{
    return plato_problem_sublist(std::forward<ParameterListType>(aParameterList))
        .sublist("Spatial Model")
        .sublist("Domains");
}

template <typename ParameterListType>
decltype(auto) domain_sublist(ParameterListType&& aParameterList, const std::string_view aDomainName)
{
    return all_domains_sublist(std::forward<ParameterListType>(aParameterList)).sublist(std::string{aDomainName});
}

template <typename ParameterListType>
decltype(auto) parameters_sublist(ParameterListType&& aParameterList)
{
    return plato_problem_sublist(std::forward<ParameterListType>(aParameterList)).sublist("Parameters");
}

template <typename ParameterListType>
decltype(auto) solver_sublist(ParameterListType&& aParameterList)
{
    return plato_problem_sublist(std::forward<ParameterListType>(aParameterList)).sublist("Linear Solver");
}

}  // namespace plato::functional

#endif
