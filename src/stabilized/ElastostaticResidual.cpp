#include "stabilized/ElastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "stabilized/ElastostaticResidual_def.hpp"
#include "stabilized/ExpInstMacros.hpp"
#include "stabilized/MechanicsElement.hpp"

PLATO_STABILIZED_EXP_INST(Plato::Stabilized::ElastostaticResidual, Plato::Stabilized::MechanicsElement)

#endif
