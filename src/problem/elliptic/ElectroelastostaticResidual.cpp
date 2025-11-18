#include "problem/elliptic/ElectroelastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ElectromechanicsElement.hpp"
#include "problem/elliptic/ElectroelastostaticResidual_def.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::ElectroelastostaticResidual, Plato::ElectromechanicsElement)

#endif
