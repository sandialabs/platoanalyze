#include "elliptic/finite_deformation_mechanics/StrainEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"
#include "elliptic/finite_deformation_mechanics/StrainEnergy_def.hpp"

PLATO_ELLIPTIC_EXP_INST(plato::elliptic::finite_deformation_mechanics::StrainEnergy, Plato::MechanicsElement)

#endif
