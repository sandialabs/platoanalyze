#include "problem/parabolic/HeatEquationResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/parabolic/ExpInstMacros.hpp"
#include "problem/parabolic/HeatEquationResidual_def.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::HeatEquationResidual, Plato::ThermalElement)

#endif
