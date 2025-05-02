#ifndef PLATO_FUNCTIONAL_FUNCTIONALINTERFACEUTILITIES_H
#define PLATO_FUNCTIONAL_FUNCTIONALINTERFACEUTILITIES_H

#include <memory>
#include <plato/analysis/AnalysisDomainMesh.hpp>
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
struct AnalysisDomainMesh;
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

/// @brief Generates an input ParameterList for the Helmholtz filter but which computes the adjoint of its Jacobian
/// in computeGradient.
/// @param aBlockNames The block names that are used in the filter, must match the blocks names in @a aMeshName.
[[nodiscard]] auto adjoint_helmholtz_filter_parameter_list(const filter::library::FilterParameters& aFilterParameters,
                                                           const std::string_view aMeshName,
                                                           const std::vector<std::string>& aBlockNames)
    -> Teuchos::ParameterList;

/// @brief Replaces the file name of the mesh in @a aParameterList with @a aMeshName.
void update_mesh_file_name(Teuchos::ParameterList& aParameterList, std::string_view aMeshName);

/// @brief Copies the nodal density field contained in @a aAnalysisDomainMesh to a ScalarVector
[[nodiscard]] Plato::ScalarVector create_control(const analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                                 const Plato::Mesh& aMesh);

/// @brief Computes a hash by combining the controls in @a aControl and the mesh coordinates in @a aMesh.
///
/// This is mainly useful for the state caching functionality.
[[nodiscard]] std::size_t hash_current_design(const Plato::ScalarVector& aControl, const Plato::Mesh& aMesh);

/// @brief Converts a ScalarVector to a std::vector and reduces to only the design variables, removing any entries
/// associated with fixed nodes.
[[nodiscard]] std::vector<double> design_variable_std_vector(
    const Plato::ScalarVector aScalarVector,
    const plato::analysis::AnalysisDomainMesh& aDesignVariables,
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
                                                           const plato::analysis::AnalysisDomainMesh& aDesignVariables,
                                                           const Plato::Mesh& aMesh,
                                                           const double aFillValue);

/// @brief Converts a AnalysisDomainMesh to a ScalarVector.
///
/// The ordering of the entries is set by the nodemap given by @a aMesh, which maps global IDs to a PA ScalarVector's
/// entries.
/// @post The size of the returned vector will be equal to the number of nodes in @a aMesh.
[[nodiscard]] Plato::ScalarVector full_nodal_scalar_vector(const plato::analysis::AnalysisDomainMesh& aDesignVariables,
                                                           const Plato::Mesh& aMesh);

/// @brief Converts a ScalarVector to the block-based data structure BlockDensities held by a AnalysisDomainMesh
/// object.
/// @param aScalarVector The vector of nodal densities to populate the result with.
/// @param aAnalysisDomainMeshIndices A AnalysisDomainMesh object whose indices will be used for the result.
/// Essentially, the density values in this object will be replaced with those in @a aScalarVector.
/// @pre All `mDesignVariableVectorIndex` entries must be less than size of @a aScalarVector.
[[nodiscard]] auto analysis_domain_mesh(Plato::ScalarVector aScalarVector,
                                        plato::analysis::AnalysisDomainMesh aDesignVariablesIndices,
                                        const Plato::Mesh& aMesh) -> plato::analysis::AnalysisDomainMesh;

/// @brief Returns the number of design variables associated with @a aAnalysisDomainMesh.
///
/// This is the max vector index found in any block in @a aAnalysisDomainMesh.
[[nodiscard]] std::size_t number_of_analysis_field_variables(
    const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh);

/// @brief Returns a std::vector cøpy of the passed-in ScalarVector.
std::vector<double> scalar_vector_to_std_vector(const Plato::ScalarVector aScalarVector);

/// @brief Converts a filter radius given in physical space to one in Helmholtz space.
[[nodiscard]] const auto helmholtz_radius_from_physical_radius(const double aPhysicalRadius) -> double;

}  // namespace plato::functional

#endif
