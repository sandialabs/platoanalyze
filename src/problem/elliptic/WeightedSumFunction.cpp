#include "problem/elliptic/WeightedSumFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/Mechanics.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "problem/elliptic/WeightedSumFunction_def.hpp"
#include "problem/elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::WeightedSumFunction,
                  plato::elliptic::finite_deformation_mechanics::FiniteDeformationMechanics)

#endif
