#include "NaturalBCs.hpp"

#include <algorithm>

namespace Plato
{
namespace detail
{
void affirmOneValidInput(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    constexpr int kNumOptions = 6;
    const std::array<bool, kNumOptions> tAllOptions = {
        aSubList.isType<Plato::Scalar>("Value"),
        aSubList.isType<std::string>("Value"),
        aSubList.isType<std::string>("Variable"),
        aSubList.isType<Teuchos::Array<Plato::Scalar>>("Values"),
        aSubList.isType<Teuchos::Array<std::string>>("Values"),
        aSubList.isType<Teuchos::Array<std::string>>("Variables")};

    if (std::count(std::begin(tAllOptions), std::end(tAllOptions), true) > 1)
    {
        std::cout << "Parameter list: \n" << aSubList << std::endl;
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: Expected one of 'Values', 'Value', 'Variable', or 'Variables' Parameter Keyword in "
            << "Parameter Sublist: '" << aName.c_str() << "', but found multiple. Check input Parameter Keywords.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    } 
    else if (std::none_of(std::begin(tAllOptions), std::end(tAllOptions), [](const bool aValue){return aValue;}))
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: Load Boundary Condition in Parameter Sublist: '"
            << aName.c_str() << "' was NOT parsed. Check input Parameter Keywords.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }
}
}
}