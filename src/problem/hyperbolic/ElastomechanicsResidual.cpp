#include "problem/hyperbolic/ElastomechanicsResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/hyperbolic/ElastomechanicsResidual_def.hpp"
#include "problem/hyperbolic/ExpInstMacros.hpp"

PLATO_HYPERBOLIC_EXP_INST(Plato::Hyperbolic::TransientMechanicsResidual, Plato::MechanicsElement)

#endif
