#include "problem/helmholtz/AdjointProblem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "Helmholtz.hpp"
#include "element/BaseExpInstMacros.hpp"
#include "problem/helmholtz/AdjointProblem_def.hpp"

PLATO_ELEMENT_DEF(Plato::Helmholtz::AdjointProblem, Plato::HelmholtzFilter)

#endif
