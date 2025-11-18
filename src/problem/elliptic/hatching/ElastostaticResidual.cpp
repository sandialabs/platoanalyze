#include "problem/elliptic/hatching/ElastostaticResidual_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "problem/elliptic/hatching/ElastostaticResidual_def.hpp"
#include "problem/elliptic/hatching/ExpInstMacros.hpp"
#include "problem/elliptic/hatching/MechanicsElement.hpp"

PLATO_ELLIPTIC_HATCHING_EXP_INST(Plato::Elliptic::Hatching::ElastostaticResidual,
                                 Plato::Elliptic::Hatching::MechanicsElement)

#endif
