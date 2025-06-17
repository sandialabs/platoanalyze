#include "hyperbolic/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "hyperbolic/Mechanics.hpp"
#include "hyperbolic/Problem_def.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::Problem, Plato::Hyperbolic::Mechanics)

#ifdef PLATO_MICROMORPHIC
#include "hyperbolic/micromorphic/MicromorphicMechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::Problem, Plato::Hyperbolic::MicromorphicMechanics)
#endif

#endif
