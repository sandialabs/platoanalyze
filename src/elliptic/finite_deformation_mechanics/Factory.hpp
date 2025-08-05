#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_FACTORY_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_FACTORY_H

#include <Teuchos_ParameterList.hpp>
#include <memory>
#include <string>

#include "AnalyzeMacros.hpp"
#include "MakeFunctions.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoUtilities.hpp"
#include "SpatialModel.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/finite_deformation_mechanics/HyperElastostaticResidual.hpp"
#include "elliptic/finite_deformation_mechanics/StrainEnergy.hpp"
#include "elliptic/finite_deformation_mechanics/StrainInvariant.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
/// @brief factory type used in Problem and Vector Function classes to construct shared pointer to base classes of PDE
/// residuals and scalar criteria.
struct FiniteDeformationMechanicsFactory
{
    /// @brief returns a shared pointer to the base class of a vector function (PDE residual).
    template <typename EvaluationType>
    auto createVectorFunction(const Plato::SpatialDomain& aSpatialDomain,
                              Plato::DataMap& aDataMap,
                              Teuchos::ParameterList& aProblemParams,
                              std::string aPDE)
        -> std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>;

    /// @brief returns a shared pointer to the base class of a scalar function (criterion for optimization).
    template <typename EvaluationType>
    auto createScalarFunction(const Plato::SpatialDomain& aSpatialDomain,
                              Plato::DataMap& aDataMap,
                              Teuchos::ParameterList& aProblemParams,
                              std::string aFuncType,
                              std::string aFuncName)
        -> std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>;
};

template <typename EvaluationType>
auto FiniteDeformationMechanicsFactory::createVectorFunction(const Plato::SpatialDomain& aSpatialDomain,
                                                             Plato::DataMap& aDataMap,
                                                             Teuchos::ParameterList& aProblemParams,
                                                             std::string aPDE)
    -> std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
{
    return Plato::makeVectorFunction<EvaluationType, HyperElastostaticResidual>(aSpatialDomain, aDataMap,
                                                                                aProblemParams, aPDE);
}

template <typename EvaluationType>
auto FiniteDeformationMechanicsFactory::createScalarFunction(const Plato::SpatialDomain& aSpatialDomain,
                                                             Plato::DataMap& aDataMap,
                                                             Teuchos::ParameterList& aProblemParams,
                                                             std::string aFuncType,
                                                             std::string aFuncName)
    -> std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
{
    const auto tLowerFuncType = Plato::tolower(aFuncType);
    if (tLowerFuncType == "strain energy")
    {
        return Plato::makeScalarFunction<EvaluationType, StrainEnergy>(aSpatialDomain, aDataMap, aProblemParams,
                                                                       aFuncName);
    }
    else if (tLowerFuncType == "strain invariant")
    {
        return Plato::makeScalarFunction<EvaluationType, StrainInvariant>(aSpatialDomain, aDataMap, aProblemParams,
                                                                          aFuncName);
    }
    else
    {
        ANALYZE_THROWERR(std::string("Criterion type '") + aFuncType +
                         std::string("' was not recognized for finite deformation mechanics."))
    }
}

}  // namespace plato::elliptic::finite_deformation_mechanics
#endif
