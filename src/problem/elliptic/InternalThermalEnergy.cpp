#include "problem/elliptic/InternalThermalEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/InternalThermalEnergy_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::InternalThermalEnergy, Plato::ThermalElement)

#endif
