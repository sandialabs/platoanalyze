#include "NaturalBC.hpp"

namespace Plato
{
namespace 
{
std::string errorMessage(const std::string& aParameterName, const std::string& aBCName, const std::string& aError)
{
    std::stringstream tMsg;
    tMsg << "Natural Boundary Condition: '" << aParameterName << "' Parameter Keyword in "
         << "Parameter Sublist: '" << aBCName << "' " << aError;
    return tMsg.str();
}
}

std::string getStringDataAndAffirmExists(
    const std::string& aParameterName, const Teuchos::ParameterList& aSublist, const std::string& aBCName)
{
    affirmExists(aParameterName, aSublist, aBCName);
    if(aSublist.isType<std::string>(aParameterName))
    {
        return aSublist.get<std::string>(aParameterName);
    }
    else
    {
        ANALYZE_THROWERR(errorMessage(aParameterName, aBCName, "is NOT of type 'string'.").c_str())
    }
}

void affirmExists(
    const std::string& aParameterName, const Teuchos::ParameterList& aSublist, const std::string& aBCName)
{
    if(!aSublist.isParameter(aParameterName))
    {
        ANALYZE_THROWERR(errorMessage(aParameterName, aBCName, "is NOT defined.").c_str())
    }
}
}
