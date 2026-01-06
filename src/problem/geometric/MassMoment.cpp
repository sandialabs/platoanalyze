#include "problem/geometric/MassMoment_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/geometric/ExpInstMacros.hpp"
#include "problem/geometric/MassMoment_def.hpp"

PLATO_GEOMETRIC_EXP_INST_2(Plato::Geometric::MassMoment, Plato::MechanicsElement)

#endif
