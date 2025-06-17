#include "elliptic/DivisionFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Electromechanics.hpp"
#include "Mechanics.hpp"
#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "elliptic/DivisionFunction_def.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
#include "stabilized/Mechanics.hpp"
#include "stabilized/Thermomechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
