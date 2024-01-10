#pragma once

#include "PlatoTypes.hpp"

#include <Kokkos_Core.hpp>

#include <cassert>

namespace Plato
{

template<int SpatialDim, typename ScalarType = Plato::Scalar>
class Rank4SkewConstant
{
public:
    Rank4SkewConstant(): c0{{0.0}} 
    {}

    KOKKOS_INLINE_FUNCTION ScalarType
    operator()
    (Plato::OrdinalType i, 
     Plato::OrdinalType j ) const 
    {
        assert(i < mNumSkwTerms);
        assert(j < mNumSkwTerms);
        return c0[i][j];
    }

protected:
    static constexpr auto mNumSkwTerms     = (SpatialDim == 3) ? 3 :
                                             ((SpatialDim == 2) ? 1 :
                                            (((SpatialDim == 1) ? 1 : 0)));

    ScalarType c0[mNumSkwTerms][mNumSkwTerms];

};

}