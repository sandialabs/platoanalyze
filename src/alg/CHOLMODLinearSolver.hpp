#pragma once

#ifdef PLATO_UMFPACK

#include <cholmod.h>

#include <memory>
#include <optional>

#include "PlatoAbstractSolver.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato::alg
{

class CHOLMODLinearSolver : public Plato::AbstractSolver
{
   public:
    CHOLMODLinearSolver(const Teuchos::ParameterList& aSolverParams,
                        Plato::LinearSystemType aLinearSystemType,
                        std::shared_ptr<Plato::MultipointConstraints> aMPCs = {});

    ~CHOLMODLinearSolver();

    CHOLMODLinearSolver(const CHOLMODLinearSolver&) = delete;
    CHOLMODLinearSolver(CHOLMODLinearSolver&&) = delete;
    CHOLMODLinearSolver& operator=(const CHOLMODLinearSolver&) = delete;
    CHOLMODLinearSolver& operator=(CHOLMODLinearSolver&&) = delete;

    /// @brief Solves `Ax = b` with `A` given by @a aA, `b` by @a aB, and `x` stored in @a aX.
    ///
    /// @pre @a aA is symmetric and either positive definite or indefinite (as given in the constructor).
    ///  If @a aA has a non-symmetric pattern, an exception is thrown. If @a aA has a symmetric pattern but
    ///  non-symmetric entries, the system is solved using only the lower diagonal.
    ///  If CHOLMODLinearSolver was constructed with SYMMETRIC_POSITIVE_DEFINITE, but @a aA is indefinite,
    ///  a cholmod may throw a floating point exception.
    void innerSolve(Plato::CrsMatrix<int> aA, Plato::ScalarVector aX, Plato::ScalarVector aB) override;

   private:
    cholmod_common mCholmodCommon;
    cholmod_factor* mCholmodFactor = nullptr;
    std::optional<std::size_t> mCurrentMatrixPatternHash;
};

}  // namespace Plato::alg

#endif
