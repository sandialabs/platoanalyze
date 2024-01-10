#pragma once

#include "PlatoTypes.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

namespace Plato
{

template<int SpatialDim>
class TensorConstant
{
public:
    TensorConstant() : c0{{0.0}} {}

    TensorConstant(Plato::Scalar aValue) : c0{{0.0}}
    {
        for (int iDim=0; iDim<SpatialDim; iDim++)
        {
            c0[iDim][iDim] = aValue;
        }
    }

    TensorConstant(Teuchos::ParameterList& aParams);

    KOKKOS_INLINE_FUNCTION Plato::Scalar
    operator()
    (Plato::OrdinalType i, 
     Plato::OrdinalType j ) const 
    {
        return c0[i][j];
    }

private:
    Plato::Scalar c0[SpatialDim][SpatialDim];
};

}