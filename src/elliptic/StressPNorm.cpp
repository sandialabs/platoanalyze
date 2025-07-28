#include "elliptic/StressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"
#include "elliptic/StressPNorm_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::StressPNorm, Plato::MechanicsElement)

#endif
