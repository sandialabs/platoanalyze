#include "problem/elliptic/hatching/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/elliptic/hatching/Mechanics.hpp"
#include "problem/elliptic/hatching/Problem_def.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::Hatching::Problem, Plato::Elliptic::Hatching::Mechanics)

#endif
