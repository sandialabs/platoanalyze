#include "problem/elliptic/ThermoelastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermomechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/ThermoelastostaticResidual_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::ThermoelastostaticResidual, Plato::ThermomechanicsElement)

#endif
