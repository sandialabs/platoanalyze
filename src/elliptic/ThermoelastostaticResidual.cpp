#include "elliptic/ThermoelastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "ThermomechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"
#include "elliptic/ThermoelastostaticResidual_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::ThermoelastostaticResidual, Plato::ThermomechanicsElement)

#endif
