//
//  AmgXConfigs.hpp
//
#ifndef AMGX_CONFIGS_HPP
#define AMGX_CONFIGS_HPP

#include <PlatoTypes.hpp>
#include <string>

namespace Plato
{
std::string configurationString(std::string configOption,
                                Plato::Scalar tol = 1e-10,
                                int maxIters = 10000,
                                bool absTolType = true);
}

#endif
