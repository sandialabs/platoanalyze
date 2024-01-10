/*
 * ComputePrincipalStresses.cpp
 *
 *  Created on: Apr 6, 2020
 */

#include "ComputePrincipalStresses.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_VMS(Plato::ComputePrincipalStresses, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEF_INC_VMS(Plato::ComputePrincipalStresses, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_VMS(Plato::ComputePrincipalStresses, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEF_INC_VMS(Plato::ComputePrincipalStresses, Plato::SimplexThermoPlasticity, 3)
#endif
