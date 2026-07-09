#ifndef PLATO_GEOMETRIC_TESTUTILITIES_MASSPROPERTIESCRITERIONUTILITIES_H
#define PLATO_GEOMETRIC_TESTUTILITIES_MASSPROPERTIESCRITERIONUTILITIES_H

#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

namespace plato::problem::geometric::test_utilities
{
[[nodiscard]] auto mass_properties_criterion(const Teuchos::Array<std::string>& aPropertyList,
                                             const Teuchos::Array<double>& aWeightsList,
                                             const Teuchos::Array<double>& aGoldValuesList,
                                             const unsigned int aPower) -> Teuchos::ParameterList;
}
#endif
