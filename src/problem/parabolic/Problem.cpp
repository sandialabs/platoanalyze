#include "problem/parabolic/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "problem/parabolic/Problem_def.hpp"

PLATO_ELEMENT_DEF(Plato::Parabolic::Problem, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Parabolic::Problem, Plato::Thermomechanics)

#endif
