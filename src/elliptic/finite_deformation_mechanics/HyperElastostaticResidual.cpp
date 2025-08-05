#include "elliptic/finite_deformation_mechanics/HyperElastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"
#include "elliptic/finite_deformation_mechanics/HyperElastostaticResidual_def.hpp"

PLATO_ELLIPTIC_EXP_INST(plato::elliptic::finite_deformation_mechanics::HyperElastostaticResidual,
                        Plato::MechanicsElement)
#endif
