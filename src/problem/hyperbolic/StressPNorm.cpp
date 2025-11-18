#include "problem/hyperbolic/StressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/hyperbolic/ExpInstMacros.hpp"
#include "problem/hyperbolic/StressPNorm_def.hpp"

PLATO_HYPERBOLIC_EXP_INST(Plato::Hyperbolic::StressPNorm, Plato::MechanicsElement)

#endif
