#ifndef PLATO_FUNCTIONAL_FUNCTIONALINTERFACEUTILITIES_H
#define PLATO_FUNCTIONAL_FUNCTIONALINTERFACEUTILITIES_H

#include <memory>
#include <string_view>
#include <vector>

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "alg/ParallelComm.hpp"

namespace Teuchos
{
class ParameterList;
}

namespace plato::core
{
struct MeshProxy;
}  // namespace plato::core

namespace plato::filter::library
{
struct FilterParameters;
}  // namespace plato::filter::library

namespace plato::functional
{
[[nodiscard]] Plato::Comm::Machine create_machine();

/// @brief Generates an input ParameterList for running the Helmholtz filter.
[[nodiscard]] Teuchos::ParameterList helmholtz_filter_parameter_list(
    const plato::filter::library::FilterParameters& aFilterParameters, const std::string_view aMeshName);

/// @brief Replaces the file name of the mesh in @a aParameterList with @a aMeshName.
void update_mesh_file_name(Teuchos::ParameterList& aParameterList, std::string_view aMeshName);

/// @brief Copies the nodal density field contained in @a aMeshProxy to a ScalarVector
[[nodiscard]] Plato::ScalarVector create_control(const core::MeshProxy& aMeshProxy, const Plato::Mesh& aMesh);

[[nodiscard]] std::size_t hash_current_design(const Plato::ScalarVector& aControl, const Plato::Mesh& aMesh);

std::vector<double> to_std_vector(const Plato::ScalarVector aScalarVector);
Plato::ScalarVector to_scalar_vector(const std::vector<double>& aVector);
}  // namespace plato::functional

#endif
