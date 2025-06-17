#include "parabolic/PhysicsScalarFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "parabolic/PhysicsScalarFunction_def.hpp"

PLATO_ELEMENT_DEF(Plato::Parabolic::PhysicsScalarFunction, Plato::Thermal);
PLATO_ELEMENT_DEF(Plato::Parabolic::PhysicsScalarFunction, Plato::Thermomechanics);

#endif
