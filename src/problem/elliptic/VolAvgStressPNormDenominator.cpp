#include "problem/elliptic/VolAvgStressPNormDenominator_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/VolAvgStressPNormDenominator_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::VolAvgStressPNormDenominator, Plato::MechanicsElement)

#endif
