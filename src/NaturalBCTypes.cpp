#include "NaturalBCTypes.hpp"
#include "PlatoUtilities.hpp"
#include "Plato_EnumTable.hpp"

namespace Plato
{
namespace 
{
const EnumTable<Neumann> kNaturalBCTable({
    {Neumann::UNIFORM_LOAD, "uniform"},
    {Neumann::VARIABLE_LOAD, "variable load"},
    {Neumann::UNIFORM_PRESSURE, "uniform pressure"},
    {Neumann::VARIABLE_PRESSURE, "variable pressure"},
    {Neumann::STEFAN_BOLTZMANN, "stefan boltzmann"}});
}

Plato::Neumann naturalBoundaryCondition(const std::string& aTypeTag)
{
    const std::string tLowerTag = Plato::tolower(aTypeTag);
    const boost::optional<Plato::Neumann> tType = kNaturalBCTable.toEnum(tLowerTag);
    if(!tType)
    {
        ANALYZE_THROWERR(std::string("Natural Boundary Condition: 'Type' Parameter Keyword: '") + tLowerTag + "' is not supported.")
    }
    return *tType;
}
}
