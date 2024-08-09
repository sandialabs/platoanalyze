#include "HelmholtzFilterInterface.hpp"

#include <Teuchos_ParameterList.hpp>
#include <plato/linear_algebra/DynamicVector.hpp>
#include <plato/mesh/MeshDesignVariables.hpp>

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
}  // namespace

HelmholtzFilterInterface::HelmholtzFilterInterface(const plato::filter::library::FilterParameters& aFilterParameters)
    : mFunctionalInterface(helmholtz_filter_parameter_list(aFilterParameters, ""))
{
}

mesh::MeshDesignVariables HelmholtzFilterInterface::filter(
    const plato::mesh::MeshDesignVariables& aMeshDesignVariables) const
{
    const auto [tSolution, tControl] = mFunctionalInterface.solveProblem(aMeshDesignVariables);

    Plato::ScalarVector tFilteredControl = filtered_control(tSolution);
    return mesh_design_variables(tFilteredControl, aMeshDesignVariables, mFunctionalInterface.mesh());
}

plato::linear_algebra::DynamicVector<double> HelmholtzFilterInterface::jacobianTimesVector(
    const plato::mesh::MeshDesignVariables& aMeshDesignVariables,
    const plato::linear_algebra::DynamicVector<double>& aV) const
{
    const auto [tSolution, tControl] = mFunctionalInterface.solveProblem(aMeshDesignVariables);

    const Plato::ScalarVector tVAsScalarVector =
        full_nodal_scalar_vector(aV.stdVector(), aMeshDesignVariables, mFunctionalInterface.mesh());
    const Plato::ScalarVector tGradient =
        mFunctionalInterface.problem().criterionGradient(tVAsScalarVector, "Helmholtz Gradient");
    return plato::linear_algebra::DynamicVector<double>(
        design_variable_std_vector(tGradient, aMeshDesignVariables, mFunctionalInterface.mesh()));
}

}  // namespace plato::functional

namespace plato
{
std::unique_ptr<filter::library::FilterInterface> plato_create_filter(const filter::library::FilterParameters& aInput)
{
    return std::make_unique<plato::functional::HelmholtzFilterInterface>(aInput);
}
}  // namespace plato