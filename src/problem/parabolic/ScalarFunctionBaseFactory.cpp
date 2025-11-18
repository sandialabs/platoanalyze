#include "problem/parabolic/ScalarFunctionBaseFactory_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/BaseExpInstMacros.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "problem/parabolic/ScalarFunctionBaseFactory_def.hpp"

PLATO_ELEMENT_DEF(Plato::Parabolic::ScalarFunctionBaseFactory, Plato::Thermal)
PLATO_ELEMENT_DEF(Plato::Parabolic::ScalarFunctionBaseFactory, Plato::Thermomechanics)

#endif
