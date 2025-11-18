#include "problem/hyperbolic/micromorphic/RelaxedMicromorphicResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "problem/hyperbolic/ExpInstMacros.hpp"
#include "problem/hyperbolic/micromorphic/MicromorphicMechanicsElement.hpp"
#include "problem/hyperbolic/micromorphic/RelaxedMicromorphicResidual_def.hpp"

PLATO_HYPERBOLIC_EXP_INST(Plato::Hyperbolic::Micromorphic::RelaxedMicromorphicResidual,
                          Plato::Hyperbolic::MicromorphicMechanicsElement)

#endif
