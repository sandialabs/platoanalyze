#include "elliptic/InternalElasticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "MechanicsElement.hpp"
#include "elliptic/ExpInstMacros.hpp"
#include "elliptic/InternalElasticEnergy_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::InternalElasticEnergy, Plato::MechanicsElement)

#endif
