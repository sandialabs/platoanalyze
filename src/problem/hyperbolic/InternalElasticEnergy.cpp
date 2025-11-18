#include "problem/hyperbolic/InternalElasticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/hyperbolic/ExpInstMacros.hpp"
#include "problem/hyperbolic/InternalElasticEnergy_def.hpp"

PLATO_HYPERBOLIC_EXP_INST(Plato::Hyperbolic::InternalElasticEnergy, Plato::MechanicsElement)

#endif
