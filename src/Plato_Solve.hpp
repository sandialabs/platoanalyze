#ifndef PLATO_SOLVE_HPP
#define PLATO_SOLVE_HPP

#include <memory>

#include "PlatoMathFunctors.hpp"
#include "PlatoStaticsTypes.hpp"
#include "alg/ParallelComm.hpp"

namespace Plato {

namespace Solve {

    /******************************************************************************//**
     * \brief Approximate solution for linear system, A x = b, by x = R^-1 b, where
     *        R is the row sum of A.
     * \param [in]     a_A Matrix, A
     * \param [in/out] a_x Solution vector, x, with initial guess
     * \param [in]     a_b Forcing vector, b
    **********************************************************************************/
    template <Plato::OrdinalType NumDofsPerNode>
    void RowSummed(
        Teuchos::RCP<Plato::CrsMatrixType> a_A, 
        Plato::ScalarVector a_x,
        Plato::ScalarVector a_b)
        {

            Plato::RowSum tRowSumFunctor(a_A);

            Plato::InverseWeight<NumDofsPerNode> tInverseWeight;

            Plato::ScalarVector tRowSum("row sum", a_x.extent(0));

            // a_x[i] 1.0/sum_j(a_A[i,j]) * a_b[i]
            auto tNumBlockRows = a_A->rowMap().extent(0) - 1;
            Kokkos::parallel_for("row sum inverse", Kokkos::RangePolicy<>(0, tNumBlockRows), KOKKOS_LAMBDA(const Plato::OrdinalType& aBlockRowOrdinal)
            {
                // compute row sum
                tRowSumFunctor(aBlockRowOrdinal, tRowSum);

                // apply inverse weight
                tInverseWeight(aBlockRowOrdinal, tRowSum, a_b, a_x, /*scale=*/1.0);
                
            });
        }

} // namespace Solve

} // namespace Plato

#endif
