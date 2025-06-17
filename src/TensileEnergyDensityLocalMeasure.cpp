/*
 * TensileEnergyDensityLocalMeasure.cpp
 *
 */
#include "TensileEnergyDensityLocalMeasure_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "MechanicsElement.hpp"
#include "TensileEnergyDensityLocalMeasure_def.hpp"
#include "elliptic/ExpInstMacros.hpp"

PLATO_ELLIPTIC_EXP_INST_2(Plato::TensileEnergyDensityLocalMeasure, Plato::MechanicsElement)

#endif
