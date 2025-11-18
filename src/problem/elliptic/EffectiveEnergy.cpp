#include "problem/elliptic/EffectiveEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/elliptic/EffectiveEnergy_def.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::EffectiveEnergy, Plato::MechanicsElement)

#endif
