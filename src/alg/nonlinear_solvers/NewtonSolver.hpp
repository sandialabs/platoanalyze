#ifndef PLATO_ALGORITHMS_NONLINEARSOLVERS_NEWTONSOLVER_H
#define PLATO_ALGORITHMS_NONLINEARSOLVERS_NEWTONSOLVER_H

#include <type_traits>

#include "BLAS1.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTypes.hpp"
#include "alg/PlatoAbstractSolver.hpp"

namespace plato::algorithms::nonlinear_solvers
{
/// @brief Solve nonlinear system of equations using Newton's method.
class NewtonSolver
{
   public:
    NewtonSolver(const Plato::OrdinalType aMaxIterations,
                 const Plato::Scalar aResidualTolerance,
                 const Plato::Scalar aIncrementTolerance,
                 const Plato::rcp<Plato::AbstractSolver>& aSolver)
        : mMaxIterations{aMaxIterations},
          mResidualTolerance{aResidualTolerance},
          mIncrementTolerance{aIncrementTolerance},
          mSolver{aSolver}
    {
    }

    /// @brief execute Newton's method.
    /// @tparam ResidualFunc callable for computing the Residual vector.
    ///         Must have the following signature:
    ///         Plato::ScalarVector(const Plato::ScalarVector& aState)
    /// @tparam JacobianFunc callable for computing the Jacobian matrix.
    ///         Must have the following signature:
    ///         Teuchos::RCP<Plato::CrsMatrixType>(const Plato::ScalarVector& aState)
    /// @tparam ApplyBCFunc callable for applying Essential boundary conditions to Residual vector and Jacobian
    /// matrix.
    ///         Must have the following signature:
    ///         void(Teuchos::RCP<Plato::CrsMatrixType>& aMatrix, Plato::ScalarVector& aVector, const Plato::Scalar
    ///         aScale)
    template <typename ResidualFunc, typename JacobianFunc, typename ApplyBCFunc>
    [[nodiscard]] bool solve(Plato::ScalarVector& aState,
                             const ResidualFunc& aComputeResidual,
                             const JacobianFunc& aComputeJacobian,
                             const ApplyBCFunc& aApplyBoundaryConditions,
                             std::ostream& aOutStream = std::cout) const;

   private:
    Plato::OrdinalType mMaxIterations;
    Plato::Scalar mResidualTolerance;
    Plato::Scalar mIncrementTolerance;
    Plato::rcp<Plato::AbstractSolver> mSolver;
};

namespace detail
{
bool norm_tolerance_is_satisfied(const Plato::ScalarVector& aVector,
                                 const Plato::Scalar aTolerance,
                                 std::ostream& aOutStream);
}

template <typename ResidualFunc, typename JacobianFunc, typename ApplyBCFunc>
bool NewtonSolver::solve(Plato::ScalarVector& aState,
                         const ResidualFunc& aComputeResidual,
                         const JacobianFunc& aComputeJacobian,
                         const ApplyBCFunc& aApplyBoundaryConditions,
                         std::ostream& aOutStream) const
{
    static_assert(std::is_invocable_r_v<Plato::ScalarVector, ResidualFunc, const Plato::ScalarVector&>,
                  "Function object ResidualFunc has wrong signature in call to NewtonSolver.solve().");
    static_assert(std::is_invocable_r_v<Teuchos::RCP<Plato::CrsMatrixType>, JacobianFunc, const Plato::ScalarVector&>,
                  "Function object JacobianFunc has wrong signature in call to NewtonSolver.solve().");
    static_assert(std::is_invocable_v<ApplyBCFunc, Teuchos::RCP<Plato::CrsMatrixType>&, Plato::ScalarVector&,
                                      const Plato::Scalar>,
                  "Function object ApplyBCFunc has wrong signature in call to NewtonSolver.solve().");

    aOutStream << "\n Iteration | Residual Norm | Increment Norm \n";
    aOutStream << "------------------------------------------ \n";
    for (Plato::OrdinalType tIteration = 0; tIteration < mMaxIterations; ++tIteration)
    {
        aOutStream << "     " << tIteration;
        aOutStream.precision(3);

        auto tResidual = aComputeResidual(aState);
        Plato::blas1::scale(-1.0, tResidual);
        auto tJacobian = aComputeJacobian(aState);

        const Plato::Scalar tScale = (tIteration == 0) ? 1.0 : 0.0;
        aApplyBoundaryConditions(tJacobian, tResidual, tScale);

        if (mMaxIterations > 1)
        {
            if (detail::norm_tolerance_is_satisfied(tResidual, mResidualTolerance, aOutStream))
            {
                aOutStream << "\n Residual norm tolerance satisfied. \n";
                return true;
            }
        }

        Plato::ScalarVector tStateIncrement("increment", aState.extent(0));
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tStateIncrement);
        mSolver->solve(*tJacobian, tStateIncrement, tResidual);
        Plato::blas1::axpy(static_cast<Plato::Scalar>(1.0), tStateIncrement, aState);

        if (mMaxIterations > 1)
        {
            if (detail::norm_tolerance_is_satisfied(tStateIncrement, mIncrementTolerance, aOutStream))
            {
                aOutStream << "\n Solution increment norm tolerance satisfied. \n";
                return true;
            }
            aOutStream << "\n";
        }
    }
    return false;
}
}  // namespace plato::algorithms::nonlinear_solvers

#endif
