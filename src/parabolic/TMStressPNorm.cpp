#include "parabolic/TMStressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "parabolic/TMStressPNorm_def.hpp"

#include "ThermomechanicsElement.hpp"
#include "parabolic/ExpInstMacros.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::TMStressPNorm, Plato::ThermomechanicsElement)

#endif
