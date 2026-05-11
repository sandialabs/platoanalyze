#ifndef SRC_PLATO_MESH_UTILITIES_HPP_
#define SRC_PLATO_MESH_UTILITIES_HPP_

#include <typeinfo>

#include "core_types/PlatoTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "utilities/Variables.hpp"

namespace Plato
{
inline void readNodeFields(Plato::MeshIO aReader,
                           Plato::OrdinalType aStepIndex,
                           Plato::FieldTags aFieldTags,
                           Plato::Variables& aVariables)
{
    auto tTags = aFieldTags.tags();
    for (auto& tTag : tTags)
    {
        auto tData = aReader->ReadNodeData(tTag, aStepIndex);
        auto tFieldName = aFieldTags.id(tTag);
        aVariables.vector(tFieldName, tData);
    }
}
}  // namespace Plato
#endif
