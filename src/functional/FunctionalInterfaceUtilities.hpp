#ifndef PLATO_FUNCTIONAL_FUNCTIONALINTERFACEUTILITIES_H
#define PLATO_FUNCTIONAL_FUNCTIONALINTERFACEUTILITIES_H

#include <memory>
#include <plato/mesh/MeshDesignVariables.hpp>
#include <string_view>
#include <vector>

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "alg/ParallelComm.hpp"

namespace Teuchos
{
class ParameterList;
}

namespace plato::mesh
{
struct MeshDesignVariables;
}  // namespace plato::mesh

namespace plato::filter::library
{
struct FilterParameters;
}  // namespace plato::filter::library

namespace plato::functional
{
[[nodiscard]] std::string first_criterion_name(const Teuchos::ParameterList& aProblem);

[[nodiscard]] Plato::Comm::Machine create_machine();

/// @brief Generates an input ParameterList for running the Helmholtz filter.
/// @param aBlockNames The block names that are used in the filter, must match the blocks names in @a aMeshName.
[[nodiscard]] Teuchos::ParameterList helmholtz_filter_parameter_list(
    const plato::filter::library::FilterParameters& aFilterParameters,
    const std::string_view aMeshName,
    const std::vector<std::string>& aBlockNames);

/// @brief Replaces the file name of the mesh in @a aParameterList with @a aMeshName.
void update_mesh_file_name(Teuchos::ParameterList& aParameterList, std::string_view aMeshName);

/// @brief Copies the nodal density field contained in @a aMeshDesignVariables to a ScalarVector
[[nodiscard]] Plato::ScalarVector create_control(const mesh::MeshDesignVariables& aMeshDesignVariables,
                                                 const Plato::Mesh& aMesh);

/// @brief Computes a hash by combining the controls in @a aControl and the mesh coordinates in @a aMesh.
///
/// This is mainly useful for the state caching functionality.
[[nodiscard]] std::size_t hash_current_design(const Plato::ScalarVector& aControl, const Plato::Mesh& aMesh);

/// @brief Converts a ScalarVector to a std::vector and reduces to only the design variables, removing any entries
/// associated with fixed nodes.
[[nodiscard]] std::vector<double> design_variable_std_vector(const Plato::ScalarVector aScalarVector,
                                                             const plato::mesh::MeshDesignVariables& aDesignVariables,
                                                             const Plato::Mesh& aMesh);

/// @brief Converts a std::vector to a ScalarVector and expands to the full set of nodes. The entries of @a aVector are
/// copied and the indices of @a aDesignVariables are used to place the entries. The missing entries are filled with @a
/// aFillValue.
///
/// Assume `N` is the number of design variables  (from @a aDesignVariables ) and `M` is the total number of nodes (from
/// @a aMesh ).
/// @pre The size of @a aVector must be `N`, the number of design variables.
/// @post The size of the returned vector will be `M`, the total number of nodes.
[[nodiscard]] Plato::ScalarVector full_nodal_scalar_vector(const std::vector<double>& aVector,
                                                           const plato::mesh::MeshDesignVariables& aDesignVariables,
                                                           const Plato::Mesh& aMesh,
                                                           const double aFillValue);

/// @brief Converts a MeshDesignVariables to a ScalarVector.
///
/// The ordering of the entries is set by the nodemap give by @a aMesh, which maps global IDs to a PA ScalarVector's
/// entries.
/// @post The size of the returned vector will be equal to the number of nodes in @a aMesh.
[[nodiscard]] Plato::ScalarVector full_nodal_scalar_vector(const plato::mesh::MeshDesignVariables& aDesignVariables,
                                                           const Plato::Mesh& aMesh);

/// @brief Converts a ScalarVector the block-based data structure BlockDensities held by a MeshDesignVariables object.
/// @param aScalarVector The vector of nodal densities to populate the result with.
/// @param aMeshDesignVariablesIndices A MeshDesignVariables object whose indices will be used for the result.
/// Essentially, the density values in this object will be replaced with those in @a aScalarVector.
/// @pre All `mDesignVariableVectorIndex` entries must be less than size of @a aScalarVector.
[[nodiscard]] auto mesh_design_variables(Plato::ScalarVector aScalarVector,
                                         plato::mesh::MeshDesignVariables aDesignVariablesIndices,
                                         const Plato::Mesh& aMesh) -> plato::mesh::MeshDesignVariables;

/// @brief Returns the number of design variables associated with @a aMeshDesignVariables.
///
/// This is the max vector index found in any block in @a aMeshDesignVariables.
std::size_t number_of_design_variables(const plato::mesh::MeshDesignVariables& aMeshDesignVariables);

}  // namespace plato::functional

#endif
