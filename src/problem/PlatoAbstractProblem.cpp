#include "problem/PlatoAbstractProblem.hpp"

#include "domain/InputDataUtils.hpp"

namespace Plato
{
auto make_data_map(const Teuchos::ParameterList& aInputs, const Plato::Mesh& aMesh) -> Plato::DataMap
{
    Plato::DataMap tDataMap;
    read_input_data(aInputs, tDataMap, aMesh);
    return tDataMap;
}

}  // namespace Plato
