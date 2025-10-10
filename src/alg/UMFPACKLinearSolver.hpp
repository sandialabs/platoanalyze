#pragma once

#ifdef PLATO_UMFPACK

#include <umfpack.h>

#include <string>
#include <vector>

#include "PlatoAbstractSolver.hpp"
#include "PlatoStaticsTypes.hpp"
#include "alg/SuiteSparseUtils.hpp"

namespace Plato::alg
{
/// @brief Interface to the UMFPACK sparse direct linear solver. May be used for any type of sparse system.
///
/// For symmetric matrices, see CHOLMODLinearSolver.
class UMFPACKLinearSolver : public Plato::AbstractSolver
{
   public:
    UMFPACKLinearSolver(const Teuchos::ParameterList &aSolverParams,
                        std::shared_ptr<Plato::MultipointConstraints> aMPCs = nullptr);

    void innerSolve(Plato::CrsMatrix<int> aA, Plato::ScalarVector aX, Plato::ScalarVector aB) override;
};

}  // namespace Plato::alg

#endif
