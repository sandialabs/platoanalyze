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
    [[nodiscard]] analysis::AnalysisDomainMesh filter(
        const analysis::AnalysisDomainMesh& aAnalysisDomainMesh) const override;

    ///@brief Evaluate the product of a row vector with the Jacobian matrix.
    [[nodiscard]] plato::linear_algebra::DynamicVector<double> rowVectorTimesJacobian(
        const analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
        const plato::linear_algebra::DynamicVector<double>& aV) const override;

    ///@brief Evaluate the product of a row vector with the transpose of the Jacobian matrix.
    [[nodiscard]] plato::linear_algebra::DynamicVector<double> rowVectorTimesAdjointJacobian(
        const analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
        const plato::linear_algebra::DynamicVector<double>& aV) const;

   private:
    plato::filter::library::FilterParameters mFilterParameters;
    mutable FunctionalInterface mFunctionalInterface;
    mutable FunctionalInterface mFunctionalInterfaceForAdjoint;
};
}  // namespace plato::functional

#endif
