#include "problem/elliptic/EMStressPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ElectromechanicsElement.hpp"
#include "problem/elliptic/EMStressPNorm_def.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::EMStressPNorm, Plato::ElectromechanicsElement)

#endif
