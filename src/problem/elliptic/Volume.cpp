#include "problem/elliptic/Volume_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "element/ThermomechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/Volume_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::Volume, Plato::ThermomechanicsElement)

#endif
