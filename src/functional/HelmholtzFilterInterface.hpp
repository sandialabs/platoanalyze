#ifndef PLATO_FUNCTIONAL_HELMHOLTZFILTERINTERFACE_H
#define PLATO_FUNCTIONAL_HELMHOLTZFILTERINTERFACE_H

#include <plato/filter/FilterInterface.hpp>

#include "FunctionalInterface.hpp"

namespace plato::functional
{
/// @brief Helmholtz filter interface to PlatoFunctional.
class HelmholtzFilterInterface : public plato::filter::library::FilterInterface
{
   public:
    /// @brief Constructor for HelmholtzFilterInterface
    explicit HelmholtzFilterInterface(const plato::filter::library::FilterParameters& aFilterParameters);

    /// @brief Perform filter operation on controls contained in the parameter
    [[nodiscard]] core::MeshProxy filter(const core::MeshProxy& aMeshProxy) const override;

    ///@brief Evaluate the jacobian times a direction vector.
    [[nodiscard]] plato::linear_algebra::DynamicVector<double> jacobianTimesVector(
        const core::MeshProxy& aMeshProxy,
        const plato::linear_algebra::DynamicVector<double>& aV) const override;

   private:
    mutable FunctionalInterface mFunctionalInterface;
};
}  // namespace plato::functional

#endif
