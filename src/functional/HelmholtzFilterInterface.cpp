#include "HelmholtzFilterInterface.hpp"

#include <Teuchos_ParameterList.hpp>

#include "DynamicVector.hpp"
#include "FunctionalInterfaceUtilities.hpp"
#include "MeshProxy.hpp"
#include "PlatoAbstractProblem.hpp"
#include "Solutions.hpp"

namespace Plato::Functional {
namespace {
Plato::ScalarVector filtered_control(const Plato::Solutions& aSolution) {
  return Kokkos::subview(aSolution.get("State"), 0, Kokkos::ALL());
}
}  // namespace

HelmholtzFilterInterface::HelmholtzFilterInterface(const Plato::Functional::FilterParameters& aFilterParameters)
    : mFunctionalInterface(helmholtz_filter_parameter_list(aFilterParameters, "")) {}

MeshProxy HelmholtzFilterInterface::filter(const MeshProxy& aMeshProxy) const {
  const auto [tSolution, tControl] = mFunctionalInterface.solveProblem(aMeshProxy);

  Plato::ScalarVector tFilteredControl = filtered_control(tSolution);
  return MeshProxy{aMeshProxy.mFileName, to_std_vector(tFilteredControl)};
}

Core::DynamicVector<double> HelmholtzFilterInterface::jacobianTimesVector(const MeshProxy& aMeshProxy,
                                                                          const Core::DynamicVector<double>& aV) const {
  const auto [tSolution, tControl] = mFunctionalInterface.solveProblem(aMeshProxy);

  const Plato::ScalarVector tVAsScalarVector = to_scalar_vector(aV.stdVector());
  const Plato::ScalarVector tGradient =
      mFunctionalInterface.problem().criterionGradient(tVAsScalarVector, "Helmholtz Gradient");
  return Core::DynamicVector<double>(to_std_vector(tGradient));
}

std::unique_ptr<FilterInterface> plato_create_filter(const FilterParameters& aInput) {
  return std::make_unique<HelmholtzFilterInterface>(aInput);
}

}  // namespace Plato::Functional
