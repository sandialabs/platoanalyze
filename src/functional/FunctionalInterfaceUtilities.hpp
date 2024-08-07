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
[[nodiscard]] Teuchos::ParameterList helmholtz_filter_parameter_list(
    const plato::filter::library::FilterParameters& aFilterParameters, const std::string_view aMeshName);

/// @brief Replaces the file name of the mesh in @a aParameterList with @a aMeshName.
void update_mesh_file_name(Teuchos::ParameterList& aParameterList, std::string_view aMeshName);

/// @brief Copies the nodal density field contained in @a aMeshDesignVariables to a ScalarVector
[[nodiscard]] Plato::ScalarVector create_control(const mesh::MeshDesignVariables& aMeshDesignVariables,
                                                 const Plato::Mesh& aMesh);

/// @brief Computes a hash by combining the controls in @a aControl and the mesh coordinates in @a aMesh.
///
/// This is mainly useful for the state cacheing functionality.
[[nodiscard]] std::size_t hash_current_design(const Plato::ScalarVector& aControl, const Plato::Mesh& aMesh);

/// @brief Converts a ScalarVector to a std::vector by copying all entries.
[[nodiscard]] std::vector<double> to_std_vector(const Plato::ScalarVector aScalarVector);

/// @brief Converts a std::vector to a ScalarVector by copying all entries.
[[nodiscard]] Plato::ScalarVector to_scalar_vector(const std::vector<double>& aVector);

/// @brief Converts a MeshDesignVariables to a ScalarVector.
///
/// The ordering of the entries is set by the `plato::mesh::Density::mDesignVariableVectorIndex` field in each Density.
/// @post The size of the returned vector will be equal to the number of nodes in @a aMesh.
[[nodiscard]] Plato::ScalarVector to_scalar_vector(const plato::mesh::MeshDesignVariables& aDesignVariables,
                                                   const Plato::Mesh& aMesh);

/// @brief Converts a ScalarVector the block-based data structure BlockDensities held by a MeshDesignVariables object.
/// @param aScalarVector The vector of nodal densities to populate the result with.
/// @param aMeshDesignVariablesIndices A MeshDesignVariables object whose indices will be used for the result.
/// Essentially, the density values in this object will be replaced with those in @a aScalarVector.
/// @pre All `mDesignVariableVectorIndex` entries must be less than size of @a aScalarVector.
[[nodiscard]] auto to_mesh_design_variables(Plato::ScalarVector aScalarVector,
                                            plato::mesh::MeshDesignVariables aMeshDesignVariablesIndices)
    -> plato::mesh::MeshDesignVariables;

}  // namespace plato::functional

#endif
