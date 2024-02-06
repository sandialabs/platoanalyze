#include "HelmholtzFilterInterface.hpp"

#include <Teuchos_ParameterList.hpp>

#include "DynamicVector.hpp"
#include "FunctionalInterfaceUtilities.hpp"
#include "MeshProxy.hpp"
#include "PlatoAbstractProblem.hpp"
#include "Solutions.hpp"

namespace plato::functional::filter::extension
{
namespace
{
Plato::ScalarVector filtered_control(const Plato::Solutions& aSolution)
{
    return Kokkos::subview(aSolution.get("State"), 0, Kokkos::ALL());
}
}  // namespace

HelmholtzFilterInterface::HelmholtzFilterInterface(const library::FilterParameters& aFilterParameters)
    : mFunctionalInterface(helmholtz_filter_parameter_list(aFilterParameters, ""))
{
}

core::MeshProxy HelmholtzFilterInterface::filter(const core::MeshProxy& aMeshProxy) const
{
    const auto [tSolution, tControl] = mFunctionalInterface.solveProblem(aMeshProxy);

    Plato::ScalarVector tFilteredControl = filtered_control(tSolution);
    return core::MeshProxy{aMeshProxy.mFileName, to_std_vector(tFilteredControl)};
}

plato::functional::linear_algebra::DynamicVector<double> HelmholtzFilterInterface::jacobianTimesVector(
    const core::MeshProxy& aMeshProxy, const plato::functional::linear_algebra::DynamicVector<double>& aV) const
{
    const auto [tSolution, tControl] = mFunctionalInterface.solveProblem(aMeshProxy);

    const Plato::ScalarVector tVAsScalarVector = to_scalar_vector(aV.stdVector());
    const Plato::ScalarVector tGradient =
        mFunctionalInterface.problem().criterionGradient(tVAsScalarVector, "Helmholtz Gradient");
    return plato::functional::linear_algebra::DynamicVector<double>(to_std_vector(tGradient));
}

}  // namespace plato::functional::filter::extension

namespace plato::functional
{
std::unique_ptr<filter::library::FilterInterface> plato_create_filter(const filter::library::FilterParameters& aInput)
{
    return std::make_unique<filter::extension::HelmholtzFilterInterface>(aInput);
}
}  // namespace plato::functional