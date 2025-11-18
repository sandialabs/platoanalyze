#include "problem/elliptic/stabilized/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/elliptic/stabilized/Mechanics.hpp"
#include "problem/elliptic/stabilized/Problem_def.hpp"
#include "problem/elliptic/stabilized/Thermomechanics.hpp"

PLATO_ELEMENT_DEF(Plato::Stabilized::Problem, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Stabilized::Problem, Plato::Stabilized::Thermomechanics)

#endif
