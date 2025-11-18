#include "problem/elliptic/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/Electromechanics.hpp"
#include "problem/Mechanics.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "problem/elliptic/Problem_def.hpp"

PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Mechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(Plato::Elliptic::Problem, Plato::Electromechanics)

#endif
