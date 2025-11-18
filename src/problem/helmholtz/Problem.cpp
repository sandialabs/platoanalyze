#include "problem/helmholtz/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/helmholtz/Helmholtz.hpp"
#include "problem/helmholtz/Problem_def.hpp"

PLATO_ELEMENT_DEF(Plato::Helmholtz::Problem, Plato::HelmholtzFilter)

#endif
