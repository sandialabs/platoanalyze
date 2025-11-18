#include "problem/elliptic/TMStressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermomechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/TMStressPNorm_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::TMStressPNorm, Plato::ThermomechanicsElement)

#endif
