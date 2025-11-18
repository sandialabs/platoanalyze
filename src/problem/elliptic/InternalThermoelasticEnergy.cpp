#include "problem/elliptic/InternalThermoelasticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermomechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/InternalThermoelasticEnergy_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::InternalThermoelasticEnergy, Plato::ThermomechanicsElement)

#endif
