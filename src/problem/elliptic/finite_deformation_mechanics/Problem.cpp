#include "problem/elliptic/finite_deformation_mechanics/Problem_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION
#include "element/BaseExpInstMacros.hpp"
#include "problem/elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"
#include "problem/elliptic/finite_deformation_mechanics/Problem_def.hpp"

PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::Problem,
                  plato::elliptic::finite_deformation_mechanics::FiniteDeformationMechanics)

#endif
