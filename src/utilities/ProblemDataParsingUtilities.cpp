#include "utilities/ProblemDataParsingUtilities.hpp"

#include <Teuchos_Array.hpp>
#include <string>
#include <vector>

namespace plato::utilities
{
void get_displacement_dof_names(const Plato::OrdinalType aNumSpatialDims, std::vector<std::string>& aDofNames)
{
    aDofNames.push_back("displacement X");
    if (aNumSpatialDims > 1)
    {
        aDofNames.push_back("displacement Y");
    }
    if (aNumSpatialDims > 2)
    {
        aDofNames.push_back("displacement Z");
    }
}

std::vector<std::string> get_plot_table(const Teuchos::ParameterList& aSublist)
{
    if (aSublist.isType<Teuchos::Array<std::string>>("Plottable"))
    {
        return aSublist.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }
    else
    {
        return std::vector<std::string>{};
    }
}
}  // namespace plato::utilities
