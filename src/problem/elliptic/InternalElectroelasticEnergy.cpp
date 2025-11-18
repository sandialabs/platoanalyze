#include "problem/elliptic/InternalElectroelasticEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ElectromechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/InternalElectroelasticEnergy_def.hpp"

PLATO_ELLIPTIC_EXP_INST(Plato::Elliptic::InternalElectroelasticEnergy, Plato::ElectromechanicsElement)

#endif
