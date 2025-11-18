#include "problem/elliptic/ThermostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/ThermostaticResidual_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::ThermostaticResidual, Plato::ThermalElement)

#endif
