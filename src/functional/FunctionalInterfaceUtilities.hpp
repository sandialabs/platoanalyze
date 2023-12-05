#ifndef PLATO_FUNCTIONAL_FUNCTIONALINTERFACEUTILITIES_H
#define PLATO_FUNCTIONAL_FUNCTIONALINTERFACEUTILITIES_H

#include "alg/ParallelComm.hpp"
#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"

#include <memory>
#include <vector>
#include <string_view>

namespace Teuchos
{
class ParameterList;
}

namespace Plato::Functional
{
struct FilterParameters;
struct MeshProxy;
}

namespace Plato::Functional
{
[[nodiscard]]
Plato::Comm::Machine create_machine();

/// @brief Generates an input ParameterList for running the Helmholtz filter.
[[nodiscard]]
Teuchos::ParameterList helmholtz_filter_parameter_list(
  const Plato::Functional::FilterParameters& aFilterParameters, const std::string_view aMeshName);

/// @brief Replaces the file name of the mesh in @a aParameterList with @a aMeshName.
void update_mesh_file_name(Teuchos::ParameterList& aParameterList, std::string_view aMeshName);

/// @brief Copies the nodal density field contained in @a aMeshProxy to a ScalarVector
[[nodiscard]]
Plato::ScalarVector create_control(const MeshProxy& aMeshProxy, const Plato::Mesh& aMesh);

[[nodiscard]]
std::size_t hash_current_design(
  const Plato::ScalarVector& aControl, 
  const Plato::Mesh& aMesh);

std::vector<double> to_std_vector(const Plato::ScalarVector aScalarVector);
Plato::ScalarVector to_scalar_vector(const std::vector<double>& aVector);
}

#endif
