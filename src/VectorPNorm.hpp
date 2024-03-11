#ifndef VECTOR_P_NORM_HPP
#define VECTOR_P_NORM_HPP

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Vector p-norm functor.

 Given a vector, compute the p-norm.
 Assumes single point integration.
 */
/******************************************************************************/
template<Plato::OrdinalType VectorLength>
class VectorPNorm
{
public:

    template<typename ResultScalarType, typename VectorScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
      Plato::OrdinalType                          aCellOrdinal,
      Plato::ScalarVectorT<ResultScalarType>      aPnorm,
      Plato::ScalarMultiVectorT<VectorScalarType> aArgVector,
      Plato::OrdinalType                          aPvalue,
      Plato::ScalarVectorT<VolumeScalarType>      aCellVolume
    ) const
    {
        // compute scalar product
        //
        aPnorm(aCellOrdinal) = 0.0;
        for(Plato::OrdinalType iTerm = 0; iTerm < VectorLength; iTerm++)
        {
            aPnorm(aCellOrdinal) += aArgVector(aCellOrdinal, iTerm) * aArgVector(aCellOrdinal, iTerm);
        }
        aPnorm(aCellOrdinal) = pow(aPnorm(aCellOrdinal), aPvalue / 2.0);
        aPnorm(aCellOrdinal) *= aCellVolume(aCellOrdinal);
    }

    template<typename ResultScalarType, typename VectorScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
      Plato::OrdinalType                           aCellOrdinal,
      Plato::ScalarVectorT<ResultScalarType>       aPnorm,
      Plato::Array<VectorLength, VectorScalarType> aArgVector,
      Plato::OrdinalType                           aPvalue,
      VolumeScalarType                             aVolume
    ) const
    {
        // compute scalar product
        //
        ResultScalarType tPnorm(0.0);
        for(Plato::OrdinalType iTerm = 0; iTerm < VectorLength; iTerm++)
        {
            tPnorm += aArgVector(iTerm) * aArgVector(iTerm);
        }
        tPnorm = pow(tPnorm, aPvalue / 2.0);
        tPnorm *= aVolume;
        Kokkos::atomic_add(&aPnorm(aCellOrdinal), tPnorm);
    }
};
// class VectorPNorm

}// namespace Plato

#endif
