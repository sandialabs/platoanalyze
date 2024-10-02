#include "helmholtz/AdjointProblem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "helmholtz/AdjointProblem_def.hpp"

#include "Helmholtz.hpp"
#include "BaseExpInstMacros.hpp"

PLATO_ELEMENT_DEF(Plato::Helmholtz::AdjointProblem, Plato::HelmholtzFilter)

#endif
