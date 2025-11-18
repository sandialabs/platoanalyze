#include "problem/elliptic/stabilized/ElastostaticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/stabilized/ElastostaticEnergy_def.hpp"
#include "problem/elliptic/stabilized/MechanicsElement.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Stabilized::ElastostaticEnergy, Plato::Stabilized::MechanicsElement)

#endif
