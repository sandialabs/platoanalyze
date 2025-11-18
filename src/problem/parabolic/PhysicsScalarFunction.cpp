#include "problem/parabolic/PhysicsScalarFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "problem/parabolic/PhysicsScalarFunction_def.hpp"

PLATO_ELEMENT_DEF(Plato::Parabolic::PhysicsScalarFunction, Plato::Thermal);
PLATO_ELEMENT_DEF(Plato::Parabolic::PhysicsScalarFunction, Plato::Thermomechanics);

#endif
