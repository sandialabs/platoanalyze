/*
 * CriterionMeanSurfaceTemperature.cpp
 *
 *  Created on: Apr 8, 2021
 */

#include "hyperbolic/fluids/CriterionMeanSurfaceTemperature.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionMeanSurfaceTemperature, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionMeanSurfaceTemperature, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionMeanSurfaceTemperature, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif
