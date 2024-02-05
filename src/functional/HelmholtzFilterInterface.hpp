#ifndef PLATO_FUNCTIONAL_HELMHOLTZFILTERINTERFACE_H
#define PLATO_FUNCTIONAL_HELMHOLTZFILTERINTERFACE_H

#include "FilterInterface.hpp"
#include "FunctionalInterface.hpp"

namespace plato::functional::filter::extension
{
/// @brief Helmholtz filter interface to PlatoFunctional.
class HelmholtzFilterInterface : public library::FilterInterface
{
   public:
    /// @brief Constructor for HelmholtzFilterInterface
    explicit HelmholtzFilterInterface(const library::FilterParameters& aFilterParameters);

    /// @brief Perform filter operation on controls contained in the parameter
    [[nodiscard]] Plato::Functional::MeshProxy filter(const Plato::Functional::MeshProxy& aMeshProxy) const override;

    ///@brief Evaluate the jacobian times a direction vector.
    [[nodiscard]] Plato::Functional::Core::DynamicVector<double> jacobianTimesVector(
        const Plato::Functional::MeshProxy& aMeshProxy,
        const Plato::Functional::Core::DynamicVector<double>& aV) const override;

   private:
    mutable Plato::Functional::FunctionalInterface mFunctionalInterface;
};
}  // namespace plato::functional::filter::extension

#endif
