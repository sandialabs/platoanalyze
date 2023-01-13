#include "NaturalBCData.hpp"

namespace Plato
{
namespace detail
{
ScalarVector getNodalData(const Plato::MeshIO& aMeshIO, const std::string& aVariableName)
{
    constexpr int kStepIndex = 0;
    if(aVariableName.empty() || aVariableName == "0")
    {
        // ScalarVector initializes data to 0
        return ScalarVector("Natural BC data", aMeshIO->NumNodes());
    }
    else 
    {
        return aMeshIO->ReadNodeData(aVariableName, kStepIndex);
    }
}
}
}
