/***************************************************************************//**
 * \brief Natural boundary condition type ENUM
*******************************************************************************/

#pragma once

#include <string>

namespace Plato
{
enum struct Neumann
{
    UNDEFINED = 0,
    UNIFORM_LOAD = 1,
    VARIABLE_LOAD = 2,
    UNIFORM_PRESSURE = 3,
    VARIABLE_PRESSURE = 4,
    STEFAN_BOLTZMANN = 5
};

/// @return The boundary condition type corresponding to @a aType
/// @throw std::runtime_error if @a aType does not match (case insensitive) 
///  one of:
///  * uniform
///  * uniform pressure
///  * variable pressure
Plato::Neumann naturalBoundaryCondition(const std::string& aType);
}
