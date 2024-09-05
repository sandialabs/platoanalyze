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
    [[nodiscard]] design_variables::MeshDesignVariables filter(
        const design_variables::MeshDesignVariables& aMeshDesignVariables) const override;

    ///@brief Evaluate the jacobian times a direction vector.
    [[nodiscard]] plato::linear_algebra::DynamicVector<double> jacobianTimesVector(
        const design_variables::MeshDesignVariables& aMeshDesignVariables,
        const plato::linear_algebra::DynamicVector<double>& aV) const override;

   private:
    plato::filter::library::FilterParameters mFilterParameters;
    mutable FunctionalInterface mFunctionalInterface;
};
}  // namespace plato::functional

#endif
