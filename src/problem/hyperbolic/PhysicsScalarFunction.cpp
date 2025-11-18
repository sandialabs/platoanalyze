#include "problem/hyperbolic/PhysicsScalarFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/hyperbolic/Mechanics.hpp"
#include "problem/hyperbolic/PhysicsScalarFunction_def.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::PhysicsScalarFunction, Plato::Hyperbolic::Mechanics)

#ifdef PLATO_MICROMORPHIC
#include "problem/hyperbolic/micromorphic/MicromorphicMechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Hyperbolic::PhysicsScalarFunction, Plato::Hyperbolic::MicromorphicMechanics)
#endif

#endif
