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
[[nodiscard]] bool affirm_input_mesh_blocks_match_mesh(const Teuchos::ParameterList& aParameterList,
                                                       const Plato::Mesh& aMesh);

/// @brief Returns an error message for any validation errors found.
[[nodiscard]] auto error_messages(const Teuchos::ParameterList& aParameterList, const Plato::Mesh& aMesh)
    -> std::string;

}  // namespace plato::functional

#endif
