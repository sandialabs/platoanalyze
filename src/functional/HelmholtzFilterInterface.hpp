#ifndef PLATO_FUNCTIONAL_HELMHOLTZFILTERINTERFACE_H
#define PLATO_FUNCTIONAL_HELMHOLTZFILTERINTERFACE_H

#include "FilterInterface.hpp"

#include "FunctionalInterface.hpp"

namespace Plato::Functional
{
/// @brief Helmholtz filter interface to PlatoFunctional.
class HelmholtzFilterInterface : public Plato::Functional::FilterInterface
{
public:
  /// @brief Constructor for HelmholtzFilterInterface
  /// @param aFilterParameters 
  explicit HelmholtzFilterInterface(const Plato::Functional::FilterParameters& aFilterParameters);

  /// @brief Perform filter operation on controls contained in the parameter
  /// @param aMeshProxy 
  /// @return a MeshProxy object of the filtered control variables 
  [[nodiscard]]
  MeshProxy filter(const MeshProxy& aMeshProxy) const override;

  ///@brief Evaluate the jacobian times a direction vector. 
  ///
  ///@param aMeshProxy 
  ///@param aV 
  ///@return ROL::StdVector<double> 
  [[nodiscard]]
  ROL::StdVector<double> jacobianTimesVector(
      const MeshProxy& aMeshProxy, const ROL::StdVector<double>& aV) const override;
private:
  mutable FunctionalInterface mFunctionalInterface;
};
}

#endif
