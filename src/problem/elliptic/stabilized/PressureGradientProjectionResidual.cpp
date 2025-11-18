#include "problem/elliptic/stabilized/PressureGradientProjectionResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "problem/elliptic/stabilized/ExpInstMacros.hpp"
#include "problem/elliptic/stabilized/MechanicsElement.hpp"
#include "problem/elliptic/stabilized/PressureGradientProjectionResidual_def.hpp"
#include "problem/elliptic/stabilized/ProjectionElement.hpp"
#include "problem/elliptic/stabilized/ThermomechanicsElement.hpp"

PLATO_STABILIZED_EXP_INST_2(Plato::Stabilized::PressureGradientProjectionResidual,
                            Plato::Stabilized::ProjectionElement,
                            Plato::Stabilized::MechanicsElement)
PLATO_STABILIZED_EXP_INST_2(Plato::Stabilized::PressureGradientProjectionResidual,
                            Plato::Stabilized::ProjectionElement,
                            Plato::Stabilized::ThermomechanicsElement)
#endif
