#include "elliptic/finite_deformation_mechanics/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION
#include "BaseExpInstMacros.hpp"
#include "elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"
#include "elliptic/finite_deformation_mechanics/Problem_def.hpp"

PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::Problem,
                  plato::elliptic::finite_deformation_mechanics::FiniteDeformationMechanics)

#endif
