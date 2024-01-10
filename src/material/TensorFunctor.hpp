#pragma once

#include "PlatoTypes.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

namespace Plato
{

template<int SpatialDim>
class TensorFunctor
{
public:
    TensorFunctor() : c0{{0.0}}, c1{{0.0}}, c2{{0.0}} {}

    TensorFunctor(Plato::Scalar aValue) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
    {
        for (int iDim=0; iDim<SpatialDim; iDim++)
        {
            c0[iDim][iDim] = aValue;
        }
    }

    TensorFunctor(Teuchos::ParameterList& aParams);

    template<typename TScalarType>
    KOKKOS_INLINE_FUNCTION TScalarType
    operator()
    (TScalarType aInput, 
     Plato::OrdinalType i, 
     Plato::OrdinalType j ) const 
    {
        TScalarType tRetVal(aInput);
        tRetVal *= c1[i][j];
        tRetVal += c0[i][j];
        tRetVal += aInput*aInput*c2[i][j];
        return tRetVal;
    }

private:
    Plato::Scalar c0[SpatialDim][SpatialDim];
    Plato::Scalar c1[SpatialDim][SpatialDim];
    Plato::Scalar c2[SpatialDim][SpatialDim];
};

}