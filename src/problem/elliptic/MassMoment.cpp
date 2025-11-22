#include "problem/elliptic/MassMoment_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "element/ThermalElement.hpp"
#include "element/ThermomechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/MassMoment_def.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::ThermomechanicsElement)

#endif
