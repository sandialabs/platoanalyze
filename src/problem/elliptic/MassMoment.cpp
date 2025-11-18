#include "problem/elliptic/MassMoment_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ElectromechanicsElement.hpp"
#include "element/MechanicsElement.hpp"
#include "element/ThermalElement.hpp"
#include "element/ThermomechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/MassMoment_def.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::ThermalElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::ThermomechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::ElectromechanicsElement)

#ifdef PLATO_STABILIZED
#include "problem/elliptic/stabilized/MechanicsElement.hpp"
#include "problem/elliptic/stabilized/ThermomechanicsElement.hpp"
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::Stabilized::MechanicsElement)
PLATO_ELLIPTIC_EXP_INST_2(Plato::Elliptic::MassMoment, Plato::Stabilized::ThermomechanicsElement)
#endif

#endif
