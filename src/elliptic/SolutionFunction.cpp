#include "elliptic/SolutionFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Electromechanics.hpp"
#include "Mechanics.hpp"
#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "elliptic/SolutionFunction_def.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::SolutionFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::SolutionFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::SolutionFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::SolutionFunction, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
#include "stabilized/Mechanics.hpp"
#include "stabilized/Thermomechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Elliptic::SolutionFunction, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::SolutionFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
