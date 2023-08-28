#pragma once

#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Tensile energy density functor.
 *  
 *  Given principal strains and lame constants, return the tensile energy density
 *  (Assumes isotropic linear elasticity. In 2D assumes plane strain!)
 */
/******************************************************************************/
template<Plato::OrdinalType mNumSpatialDims>
class TensileEnergyDensity
{
  public:

    template<typename StrainType, typename ResultType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
              Plato::OrdinalType aCellOrdinal,
        const Plato::Array<mNumSpatialDims, StrainType> & aPrincipalStrains,
        const Plato::Scalar & aLameLambda,
        const Plato::Scalar & aLameMu,
        const ResultType & aWeightTimesDet,
        const Plato::ScalarVectorT<ResultType> & aTensileEnergyDensity
    ) const 
    {
        ResultType tTensileEnergyDensity = static_cast<Plato::Scalar>(0.0);
        StrainType tStrainTrace = static_cast<Plato::Scalar>(0.0);
        for (Plato::OrdinalType tDim = 0; tDim < mNumSpatialDims; ++tDim)
        {
            tStrainTrace += aPrincipalStrains(tDim);
            if (aPrincipalStrains(tDim) >= 0.0)
            {
                tTensileEnergyDensity += (aPrincipalStrains(tDim) * 
                                          aPrincipalStrains(tDim) * aLameMu);
            }
        }
        StrainType tStrainTraceTensile = (tStrainTrace >= 0.0) ? tStrainTrace : static_cast<Plato::Scalar>(0.0);
        tTensileEnergyDensity += (aLameLambda * tStrainTraceTensile * 
                                                tStrainTraceTensile * static_cast<Plato::Scalar>(0.5));
        Kokkos::atomic_add(&aTensileEnergyDensity(aCellOrdinal), aWeightTimesDet*tTensileEnergyDensity);
    }
};
//class TensileEnergyDensity


}
//namespace Plato
