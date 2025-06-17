#include "helmholtz/AdjointProblem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Helmholtz.hpp"
#include "helmholtz/AdjointProblem_def.hpp"

PLATO_ELEMENT_DEF(Plato::Helmholtz::AdjointProblem, Plato::HelmholtzFilter)

#endif
