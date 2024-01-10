#pragma once

#include "VoigtMap.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato {
  template<int SpaceDim, typename T>
  KOKKOS_INLINE_FUNCTION
  Plato::Matrix<SpaceDim,SpaceDim,T>
  FromVoigt(
    Plato::ScalarArray3DT<T> aVoigtTensor, 
    Plato::OrdinalType       aOrdinal_0,
    Plato::OrdinalType       aOrdinal_1,
    Plato::Scalar            aShearFactor=1.0
  )
  {
    Plato::VoigtMap<SpaceDim> tVoigtMap;
    Plato::Matrix<SpaceDim,SpaceDim,T> tTensor;
    for(int i=0; i<SpaceDim; i++)
      for(int j=0; j<SpaceDim; j++)
      {
        Plato::Scalar tShearFactor = (i!=j) ? aShearFactor : 1.0;
        tTensor(i,j) = tShearFactor*aVoigtTensor(aOrdinal_0, aOrdinal_1, tVoigtMap.V[i][j]);
      }
    return tTensor;
  }

  template<int SpaceDim, typename T>
  KOKKOS_INLINE_FUNCTION
  void
  ToVoigt(
    Plato::Matrix<SpaceDim,SpaceDim,T> const & aTensor,
    Plato::ScalarArray3DT<T>                   aVoigtTensor, 
    Plato::OrdinalType                         aOrdinal_0,
    Plato::OrdinalType                         aOrdinal_1,
    Plato::Scalar                              aShearFactor=1.0
  )
  {
    Plato::VoigtMap<SpaceDim> tVoigtMap;
    for(int i=0; i<VoigtMap<SpaceDim>::cNumVoigtTerms; i++)
    {
      Plato::Scalar tShearFactor = (tVoigtMap.I[i] != tVoigtMap.J[i]) ? aShearFactor : 1.0;
      aVoigtTensor(aOrdinal_0, aOrdinal_1, i) = tShearFactor*aTensor(tVoigtMap.I[i],tVoigtMap.J[i]);
    }
  }
}
