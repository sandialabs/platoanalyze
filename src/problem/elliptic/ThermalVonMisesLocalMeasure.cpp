/*
 * ThermalVonMisesLocalMeasure.cpp
 *
 */

#include "problem/elliptic/ThermalVonMisesLocalMeasure_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermomechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/ThermalVonMisesLocalMeasure_def.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::ThermalVonMisesLocalMeasure, Plato::ThermomechanicsElement)

#endif
