#include "problem/elliptic/stabilized/ThermoelastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "problem/elliptic/stabilized/ExpInstMacros.hpp"
#include "problem/elliptic/stabilized/ThermoelastostaticResidual_def.hpp"
#include "problem/elliptic/stabilized/ThermomechanicsElement.hpp"

PLATO_STABILIZED_EXP_INST(Plato::Stabilized::ThermoelastostaticResidual, Plato::Stabilized::ThermomechanicsElement)

#endif
