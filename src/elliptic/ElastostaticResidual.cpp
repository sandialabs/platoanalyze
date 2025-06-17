#include "elliptic/ElastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "MechanicsElement.hpp"
#include "elliptic/ElastostaticResidual_def.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::ElastostaticResidual, Plato::MechanicsElement)

#endif
