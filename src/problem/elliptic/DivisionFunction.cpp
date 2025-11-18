#include "problem/elliptic/DivisionFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/Electromechanics.hpp"
#include "problem/Mechanics.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "problem/elliptic/DivisionFunction_def.hpp"
#include "problem/elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction,
                  plato::elliptic::finite_deformation_mechanics::FiniteDeformationMechanics)

#ifdef PLATO_STABILIZED
#include "problem/elliptic/stabilized/Mechanics.hpp"
#include "problem/elliptic/stabilized/Thermomechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::DivisionFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
