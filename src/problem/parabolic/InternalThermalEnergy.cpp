#include "problem/parabolic/InternalThermalEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/parabolic/ExpInstMacros.hpp"
#include "problem/parabolic/InternalThermalEnergy_def.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::InternalThermalEnergy, Plato::ThermalElement)

#endif
