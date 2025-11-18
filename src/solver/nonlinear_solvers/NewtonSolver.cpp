#include "solver/nonlinear_solvers/NewtonSolver.hpp"

#include <ostream>

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace plato::algorithms::nonlinear_solvers
{
namespace detail
{
bool norm_tolerance_is_satisfied(const Plato::ScalarVector& aVector,
                                 const Plato::Scalar aTolerance,
                                 std::ostream& aOutStream)
{
    const auto tNorm = Plato::blas1::norm(aVector);
    aOutStream << std::scientific << "        " << tNorm;
    if (tNorm < aTolerance)
    {
        return true;
    }
    return false;
}
}  // namespace detail
}  // namespace plato::algorithms::nonlinear_solvers
