#include "problem/parabolic/TemperatureIntegral_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/parabolic/ExpInstMacros.hpp"
#include "problem/parabolic/TemperatureIntegral_def.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::TemperatureIntegral, Plato::ThermalElement)

#endif
