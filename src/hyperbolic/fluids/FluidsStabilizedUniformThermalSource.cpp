/*
 * FluidsStabilizedUniformThermalSource.hpp
 *
 *  Created on: June 17, 2021
 */

#include "hyperbolic/fluids/FluidsStabilizedUniformThermalSource.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::StabilizedUniformThermalSource, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::SIMP::StabilizedUniformThermalSource, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::StabilizedUniformThermalSource, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::SIMP::StabilizedUniformThermalSource, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::StabilizedUniformThermalSource, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::SIMP::StabilizedUniformThermalSource, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
#endif
