#include "elliptic/ThermostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "ThermalElement.hpp"
#include "elliptic/ExpInstMacros.hpp"
#include "elliptic/ThermostaticResidual_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::ThermostaticResidual, Plato::ThermalElement)

#endif
