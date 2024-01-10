/*
 * Plato_LocalMeasureFactory.cpp
 *
 *  Created on: Apr 2, 2019
 */

#include "LocalMeasureFactory.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF2(Plato::LocalMeasureFactory, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF2(Plato::LocalMeasureFactory, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF2(Plato::LocalMeasureFactory, Plato::SimplexMechanics, 3)
#endif