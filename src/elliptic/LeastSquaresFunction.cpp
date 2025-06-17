#include "elliptic/LeastSquaresFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "BaseExpInstMacros.hpp"
#include "Electromechanics.hpp"
#include "Mechanics.hpp"
#include "Thermal.hpp"
#include "Thermomechanics.hpp"
#include "elliptic/LeastSquaresFunction_def.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Electromechanics)

#ifdef PLATO_STABILIZED
#include "stabilized/Mechanics.hpp"
#include "stabilized/Thermomechanics.hpp"
PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::LeastSquaresFunction, Plato::Stabilized::Thermomechanics)
#endif

#endif
