#include "elliptic/PhysicsScalarFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Electromechanics.hpp"
#include "Mechanics.hpp"
#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "elliptic/PhysicsScalarFunction_def.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::PhysicsScalarFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::PhysicsScalarFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::PhysicsScalarFunction, Plato::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::PhysicsScalarFunction, Plato::Thermomechanics)

#ifdef PLATO_STABILIZED
#include "stabilized/Mechanics.hpp"
#include "stabilized/Thermomechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Elliptic::PhysicsScalarFunction, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::PhysicsScalarFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
