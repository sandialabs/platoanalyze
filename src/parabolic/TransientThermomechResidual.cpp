#include "parabolic/TransientThermomechResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "ThermomechanicsElement.hpp"
#include "parabolic/ExpInstMacros.hpp"
#include "parabolic/TransientThermomechResidual_def.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::TransientThermomechResidual, Plato::ThermomechanicsElement)

#endif
