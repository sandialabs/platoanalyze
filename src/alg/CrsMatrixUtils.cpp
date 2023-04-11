#include "alg/CrsMatrixUtils.hpp"

namespace Plato
{

void sort_matrix_column_ordinals
(Plato::OrdinalVector & tOffs,
 Plato::OrdinalVector & tOrds)
{
    auto tNumRows = tOffs.size() - 1;
    Kokkos::parallel_for("sort ordinals", Kokkos::RangePolicy<>(0, tNumRows), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
    {
        const auto tFrom = tOffs(aNodeOrdinal);
        const auto tTo = tOffs(aNodeOrdinal+1)-1;
        for( auto tIndexI=tFrom; tIndexI<tTo; tIndexI++ )
        {
            for( auto tIndexJ=tFrom; tIndexJ<tTo; tIndexJ++ )
            {
                if( tOrds(tIndexJ) > tOrds(tIndexJ+1) )
                {
                    const auto tHereHoldThis = tOrds(tIndexJ+1);
                    tOrds(tIndexJ+1) = tOrds(tIndexJ);
                    tOrds(tIndexJ) = tHereHoldThis;
                }
            }
        }
    });
}

}
