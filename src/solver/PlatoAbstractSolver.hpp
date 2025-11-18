#pragma once

#include <memory>

#include "solver/multipoint_constraint/MultipointConstraints.hpp"

namespace Plato
{

template <typename ClassT>
using rcp = std::shared_ptr<ClassT>;

enum class LinearSystemType
{
    SYMMETRIC_POSITIVE_DEFINITE,
    SYMMETRIC_INDEFINITE,
    SYMMETRIC_PATTERN
};

/******************************************************************************/
/**
 * \brief Abstract solver interface

  Note that the solve() function takes 'native' matrix and vector types.  A next
  step would be to adopt generic matrix and vector interfaces that we can wrap
  around Tpetra types, Kokkos view-based types, etc.
**********************************************************************************/
class AbstractSolver
{
   protected:
    std::shared_ptr<Plato::MultipointConstraints> mSystemMPCs = nullptr;
    Plato::Scalar mAlpha = 0.0;

    AbstractSolver() = default;
    AbstractSolver(const Teuchos::ParameterList& aSolverParams);

    virtual void innerSolve(Plato::CrsMatrix<Plato::OrdinalType> aA,
                            Plato::ScalarVector aX,
                            Plato::ScalarVector aB) = 0;

   public:
    AbstractSolver(const Teuchos::ParameterList& aSolverParams, std::shared_ptr<Plato::MultipointConstraints> aMPCs);
    virtual ~AbstractSolver() = default;

    AbstractSolver(const AbstractSolver&) = delete;
    AbstractSolver(AbstractSolver&&) = delete;
    AbstractSolver& operator=(const AbstractSolver&) = delete;
    AbstractSolver& operator=(AbstractSolver&&) = delete;

    void solve(Plato::CrsMatrix<int> aAf, Plato::ScalarVector aX, Plato::ScalarVector aB, bool aAdjointFlag = false);
};
}  // end namespace Plato
