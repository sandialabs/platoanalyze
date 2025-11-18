#include "problem/elliptic/stabilized/ElastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "problem/elliptic/stabilized/ElastostaticResidual_def.hpp"
#include "problem/elliptic/stabilized/ExpInstMacros.hpp"
#include "problem/elliptic/stabilized/MechanicsElement.hpp"

PLATO_STABILIZED_EXP_INST(Plato::Stabilized::ElastostaticResidual, Plato::Stabilized::MechanicsElement)

#endif
