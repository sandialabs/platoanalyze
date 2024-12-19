#ifndef PLATO_FUNCTIONAL_INPUTVALIDATION_H
#define PLATO_FUNCTIONAL_INPUTVALIDATION_H

#include "PlatoMesh.hpp"

namespace Teuchos
{
class ParameterList;
}

namespace plato::functional
{
/// @brief Returns `true` if all blocks in @a aMesh have a corresponding block definition in @a aParameterList.
/// @note The opposite does not have to be true, that each block in the input has a matching block in the mesh.
[[nodiscard]] bool affirm_mesh_blocks_match_input(const Teuchos::ParameterList& aParameterList,
                                                  const Plato::Mesh& aMesh);

/// @brief Returns an error message for any validation errors found.
[[nodiscard]] auto error_messages(const Teuchos::ParameterList& aParameterList, const Plato::Mesh& aMesh)
    -> std::string;

}  // namespace plato::functional

#endif
