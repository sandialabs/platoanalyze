/*
 * VonMisesLocalMeasure.cpp
 *
 */

#include "problem/elliptic/VonMisesLocalMeasure_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/VonMisesLocalMeasure_def.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::VonMisesLocalMeasure, Plato::MechanicsElement)

#endif
