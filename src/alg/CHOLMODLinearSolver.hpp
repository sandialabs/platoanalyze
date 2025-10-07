#pragma once

#ifdef PLATO_UMFPACK

#include <cholmod.h>

#include <memory>

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

    void innerSolve(Plato::CrsMatrix<int> aA, Plato::ScalarVector aX, Plato::ScalarVector aB) override;

   private:
    cholmod_common mCholmodCommon;
    cholmod_factor* mCholmodFactor = nullptr;
};

}  // namespace Plato::alg

#endif
