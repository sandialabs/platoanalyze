#include "alg/CrsMatrixUtils.hpp"

namespace Plato
{

void sort_matrix_column_ordinals
(Plato::OrdinalVector & tOffs,
 Plato::OrdinalVector & tOrds)
{
    auto tNumRows = tOffs.size() - 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
    {
        auto tFrom = tOffs(aNodeOrdinal);
        auto tTo = tOffs(aNodeOrdinal+1)-1;
        for( decltype(tFrom) tIndexI=tFrom; tIndexI<tTo; tIndexI++ )
        {
            for( decltype(tFrom) tIndexJ=tFrom; tIndexJ<tTo; tIndexJ++ )
            {
                if( tOrds(tIndexJ) > tOrds(tIndexJ+1) )
                {
                    auto tHereHoldThis = tOrds(tIndexJ+1);
                    tOrds(tIndexJ+1) = tOrds(tIndexJ);
                    tOrds(tIndexJ) = tHereHoldThis;
                }
            }
        }
    }, "sort ordinals");
}

}