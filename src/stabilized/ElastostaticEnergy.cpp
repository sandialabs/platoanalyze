#include "stabilized/ElastostaticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "elliptic/ExpInstMacros.hpp"
#include "stabilized/ElastostaticEnergy_def.hpp"
#include "stabilized/MechanicsElement.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Stabilized::ElastostaticEnergy, Plato::Stabilized::MechanicsElement)

#endif
