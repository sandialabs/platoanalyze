#include "problem/elliptic/VolumeIntegralCriterion_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/VolumeIntegralCriterion_def.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::VolumeIntegralCriterion, Plato::MechanicsElement)

#endif
