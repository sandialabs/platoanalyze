#pragma once

#include <memory>

#include "mesh/EngineMesh.hpp"
#include "mesh/EngineMeshIO.hpp"

namespace Plato
{
using Mesh = std::shared_ptr<Plato::EngineMesh>;

namespace MeshFactory
{
inline void initialize(int& aArgc, char**& aArgv) {}
inline Plato::Mesh create(std::string aFilePath) { return std::make_shared<Plato::EngineMesh>(aFilePath); }
inline void finalize() {}
}  // namespace MeshFactory
// end namespace MeshFactory

using MeshIO = std::shared_ptr<Plato::EngineMeshIO>;
namespace MeshIOFactory
{
/// @pre @a aMesh must not be `nullptr`. Checked with an assertion.
inline Plato::MeshIO create(std::string aFilePath, Plato::Mesh aMesh, std::string aMode)
{
    assert(aMesh);
    return std::make_shared<Plato::EngineMeshIO>(aFilePath, *aMesh, aMode);
}
}  // namespace MeshIOFactory
// end namespace MeshIOFactory

}  // end namespace Plato
