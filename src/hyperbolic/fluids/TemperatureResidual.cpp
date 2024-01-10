/*
 * TemperatureResidual.cpp
 *
 *  Created on: Apr 8, 2021
 */

#include "hyperbolic/fluids/TemperatureResidual.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::TemperatureResidual, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::TemperatureResidual, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::TemperatureResidual, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
#endif
