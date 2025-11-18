#include "problem/elliptic/FluxPNorm_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/FluxPNorm_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::FluxPNorm, Plato::ThermalElement)

#endif
