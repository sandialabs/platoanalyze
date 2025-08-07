#ifndef PLATO_UTILITIES_TEUCHOSPARSINGUTILITIES_H
#define PLATO_UTILITIES_TEUCHOSPARSINGUTILITIES_H

#include <Teuchos_ParameterList.hpp>
#include <optional>
#include <string>
#include <vector>

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "PlatoTypes.hpp"

namespace plato::utilities
{
/// @brief adds displacement DOF names (i.e. displacement X, ...) to the vector @param aDofNames
void get_displacement_dof_names(const Plato::OrdinalType aNumSpatialDims, std::vector<std::string>& aDofNames);

/// @brief looks for "Body Loads" in parameter list @param aProblemParams and returns a std::optional containing an
/// instance of BodyLoads if found and std::nullopt if not
template <typename EvaluationType, typename ElementType>
[[nodiscard]] auto get_body_loads(Teuchos::ParameterList& aProblemParams)
    -> std::optional<Plato::BodyLoads<EvaluationType, ElementType>>;

/// @brief looks for a list with name @param aSublistName in parameter list @param aProblemParams and returns a
/// std::optional containing an instance of NaturalBCs if found and std::nullopt if not
template <typename ElementType,
          Plato::OrdinalType NumDofs = ElementType::mNumSpatialDims,
          Plato::OrdinalType DofsPerNode = NumDofs,
          Plato::OrdinalType DofOffset = 0>
[[nodiscard]] auto get_boundary_loads(Teuchos::ParameterList& aProblemParams, const std::string& aSublistName)
    -> std::optional<Plato::NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>>;

/// @brief looks for "Plottable" in parameter list @param aProblemParams and returns a
/// std::vector containing the strings of fields to be plotted. Returns an empty vector if not found
[[nodiscard]] std::vector<std::string> get_plot_table(const Teuchos::ParameterList& aProblemParams);

template <typename EvaluationType, typename ElementType>
auto get_body_loads(Teuchos::ParameterList& aProblemParams)
    -> std::optional<Plato::BodyLoads<EvaluationType, ElementType>>
{
    if (aProblemParams.isSublist("Body Loads"))
    {
        return std::optional<Plato::BodyLoads<EvaluationType, ElementType>>(aProblemParams.sublist("Body Loads"));
    }
    else
    {
        return std::nullopt;
    }
}

template <typename ElementType,
          Plato::OrdinalType NumDofs,
          Plato::OrdinalType DofsPerNode,
          Plato::OrdinalType DofOffset>
auto get_boundary_loads(Teuchos::ParameterList& aProblemParams, const std::string& aSublistName)
    -> std::optional<Plato::NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>>
{
    if (aProblemParams.isSublist(aSublistName))
    {
        return std::optional<Plato::NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>>(
            aProblemParams.sublist(aSublistName));
    }
    else
    {
        return std::nullopt;
    }
}
}  // namespace plato::utilities
#endif
