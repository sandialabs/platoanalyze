#include "problem/parabolic/TemperatureAverage_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/parabolic/ExpInstMacros.hpp"
#include "problem/parabolic/TemperatureAverage_def.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::TemperatureAverage, Plato::ThermalElement)

#endif
