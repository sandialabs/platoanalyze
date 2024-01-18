#ifndef PLATO_FUNCTIONAL_HELMHOLTZFILTERINTERFACE_H
#define PLATO_FUNCTIONAL_HELMHOLTZFILTERINTERFACE_H

#include "FilterInterface.hpp"
#include "FunctionalInterface.hpp"

namespace Plato::Functional {
/// @brief Helmholtz filter interface to PlatoFunctional.
class HelmholtzFilterInterface : public Plato::Functional::FilterInterface {
 public:
  /// @brief Constructor for HelmholtzFilterInterface
  explicit HelmholtzFilterInterface(const Plato::Functional::FilterParameters& aFilterParameters);

  /// @brief Perform filter operation on controls contained in the parameter
  [[nodiscard]] MeshProxy filter(const MeshProxy& aMeshProxy) const override;

  ///@brief Evaluate the jacobian times a direction vector.
  [[nodiscard]] Core::DynamicVector<double> jacobianTimesVector(const MeshProxy& aMeshProxy,
                                                                const Core::DynamicVector<double>& aV) const override;

 private:
  mutable FunctionalInterface mFunctionalInterface;
};
}  // namespace Plato::Functional

#endif
