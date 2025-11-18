#include "problem/elliptic/hatching/InternalElasticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "problem/elliptic/hatching/ExpInstMacros.hpp"
#include "problem/elliptic/hatching/InternalElasticEnergy_def.hpp"
#include "problem/elliptic/hatching/MechanicsElement.hpp"

PLATO_ELLIPTIC_HATCHING_EXP_INST(Plato::Elliptic::Hatching::InternalElasticEnergy,
                                 Plato::Elliptic::Hatching::MechanicsElement)

#endif
