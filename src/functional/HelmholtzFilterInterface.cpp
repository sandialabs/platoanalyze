#include "HelmholtzFilterInterface.hpp"

#include <Teuchos_ParameterList.hpp>
#include <plato/analysis/AnalysisDomainMesh.hpp>
#include <plato/linear_algebra/DynamicVector.hpp>

#include "FunctionalInterfaceUtilities.hpp"
#include "PlatoAbstractProblem.hpp"
#include "Solutions.hpp"

namespace plato::functional
{
namespace
{
Plato::ScalarVector filtered_control(const Plato::Solutions& aSolution)
{
    return Kokkos::subview(aSolution.get("State"), 0, Kokkos::ALL());
}

auto parameter_list_updater(const plato::filter::library::FilterParameters& aFilterParameters)
{
    return [tFilterParameters = aFilterParameters](const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                                   const Plato::Mesh& aMesh)
    {
        return helmholtz_filter_parameter_list(tFilterParameters, aAnalysisDomainMesh.mFileName.string(),
                                               aMesh->GetElementBlockNames());
    };
}

auto parameter_list_updater_for_adjoint(const plato::filter::library::FilterParameters& aFilterParameters)
{
    return [tFilterParameters = aFilterParameters](const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                                   const Plato::Mesh& aMesh)
    {
        return adjoint_helmholtz_filter_parameter_list(tFilterParameters, aAnalysisDomainMesh.mFileName.string(),
                                                       aMesh->GetElementBlockNames());
    };
}

}  // namespace

HelmholtzFilterInterface::HelmholtzFilterInterface(const plato::filter::library::FilterParameters& aFilterParameters)
    : mFilterParameters{aFilterParameters}
{
}

analysis::AnalysisDomainMesh HelmholtzFilterInterface::filter(
    const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh) const
{
    const auto [tSolution, tControl] =
        mFunctionalInterface.solveProblem(aAnalysisDomainMesh, parameter_list_updater(mFilterParameters));
    Plato::ScalarVector tFilteredControl = filtered_control(tSolution);
    return analysis_domain_mesh(tFilteredControl, aAnalysisDomainMesh, mFunctionalInterface.mesh());
}

plato::linear_algebra::DynamicVector<double> HelmholtzFilterInterface::rowVectorTimesJacobian(
    const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
    const plato::linear_algebra::DynamicVector<double>& aV) const
{
    const auto [tSolution, tControl] =
        mFunctionalInterface.solveProblem(aAnalysisDomainMesh, parameter_list_updater(mFilterParameters));

    constexpr auto tFixedValueGradient = 0.0;
    const Plato::ScalarVector tVAsScalarVector =
        full_nodal_scalar_vector(aV.stdVector(), aAnalysisDomainMesh, mFunctionalInterface.mesh(), tFixedValueGradient);

    const Plato::ScalarVector tGradient =
        mFunctionalInterface.problem().criterionGradient(tVAsScalarVector, "Helmholtz Gradient");

    return plato::linear_algebra::DynamicVector<double>(
        design_variable_std_vector(tGradient, aAnalysisDomainMesh, mFunctionalInterface.mesh()));
}

plato::linear_algebra::DynamicVector<double> HelmholtzFilterInterface::rowVectorTimesAdjointJacobian(
    const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
    const plato::linear_algebra::DynamicVector<double>& aV) const
{
    const auto [tSolution, tControl] = mFunctionalInterfaceForAdjoint.solveProblem(
        aAnalysisDomainMesh, parameter_list_updater_for_adjoint(mFilterParameters));

    constexpr auto tFixedValueGradient = 0.0;
    const Plato::ScalarVector tVAsScalarVector = full_nodal_scalar_vector(
        aV.stdVector(), aAnalysisDomainMesh, mFunctionalInterfaceForAdjoint.mesh(), tFixedValueGradient);

    const Plato::ScalarVector tGradient =
        mFunctionalInterfaceForAdjoint.problem().criterionGradient(tVAsScalarVector, "Helmholtz Gradient");

    return plato::linear_algebra::DynamicVector<double>(
        design_variable_std_vector(tGradient, aAnalysisDomainMesh, mFunctionalInterfaceForAdjoint.mesh()));
}

}  // namespace plato::functional

namespace plato
{
std::unique_ptr<filter::library::FilterInterface> plato_create_filter(const filter::library::FilterParameters& aInput)
{
    return std::make_unique<plato::functional::HelmholtzFilterInterface>(aInput);
}
}  // namespace plato