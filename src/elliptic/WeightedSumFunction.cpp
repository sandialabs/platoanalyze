#include "elliptic/WeightedSumFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Electromechanics.hpp"
#include "Mechanics.hpp"
#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "elliptic/WeightedSumFunction_def.hpp"
#include "elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Electromechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction,
                  plato::elliptic::finite_deformation_mechanics::FiniteDeformationMechanics)

#ifdef PLATO_STABILIZED
#include "stabilized/Mechanics.hpp"
#include "stabilized/Thermomechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
