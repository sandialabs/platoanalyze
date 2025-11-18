/*
 * TensileEnergyDensityLocalMeasure.cpp
 *
 */
#include "problem/elliptic/TensileEnergyDensityLocalMeasure_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/MechanicsElement.hpp"
#include "problem/elliptic/ExpInstMacros.hpp"
#include "problem/elliptic/TensileEnergyDensityLocalMeasure_def.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::TensileEnergyDensityLocalMeasure, Plato::MechanicsElement)

#endif
