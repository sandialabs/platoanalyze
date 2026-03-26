#include "problem/parabolic/TimeIntegratedStateAverageFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "problem/parabolic/TimeIntegratedStateAverageFunction_def.hpp"

PLATO_ELEMENT_DEF(plato::parabolic::TimeIntegratedStateAverageFunction, Plato::Thermal);
PLATO_ELEMENT_DEF(plato::parabolic::TimeIntegratedStateAverageFunction, Plato::Thermomechanics);

#endif
