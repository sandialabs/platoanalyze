/*
 * VonMisesLocalMeasure.cpp
 *
 */

#include "VonMisesLocalMeasure_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "MechanicsElement.hpp"
#include "VonMisesLocalMeasure_def.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::VonMisesLocalMeasure, Plato::MechanicsElement)

#endif
