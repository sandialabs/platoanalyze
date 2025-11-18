#include "problem/parabolic/TransientThermomechResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermomechanicsElement.hpp"
#include "problem/parabolic/ExpInstMacros.hpp"
#include "problem/parabolic/TransientThermomechResidual_def.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::TransientThermomechResidual, Plato::ThermomechanicsElement)

#endif
