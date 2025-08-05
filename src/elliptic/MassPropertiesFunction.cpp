#include "elliptic/MassPropertiesFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Electromechanics.hpp"
#include "Mechanics.hpp"
#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "elliptic/MassPropertiesFunction_def.hpp"
#include "elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction,
                  plato::elliptic::finite_deformation_mechanics::FiniteDeformationMechanics)

#ifdef PLATO_STABILIZED
#include "stabilized/Mechanics.hpp"
#include "stabilized/Thermomechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::MassPropertiesFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
