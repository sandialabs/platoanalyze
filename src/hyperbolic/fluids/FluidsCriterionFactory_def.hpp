/*
 * FluidsCriterionFactory_def.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "PlatoUtilities.hpp"

#include "hyperbolic/fluids/FluidsScalarFunction.hpp"
#include "hyperbolic/fluids/FluidsWeightedScalarFunction.hpp"
#include "hyperbolic/fluids/FluidsLeastSquaresScalarFunction.hpp"

#include "hyperbolic/fluids/FluidsCriterionFactory.hpp"

namespace Plato
{

namespace Fluids
{

template <typename PhysicsT>
std::shared_ptr<Plato::Fluids::CriterionBase>
CriterionFactory<PhysicsT>::createCriterion
(const Plato::SpatialModel    & aModel,
       Plato::DataMap         & aDataMap,
       Teuchos::ParameterList & aInputs,
       std::string            & aTag)
{
    auto tCriteriaParam = aInputs.sublist("Criteria");
    if(tCriteriaParam.isSublist(aTag) == false) 
    {
        ANALYZE_THROWERR(std::string("Parameter list block with tag '") + aTag + "' is not defined")
    }       
    auto tFunctionTag = tCriteriaParam.sublist(aTag);
    auto tType = tFunctionTag.get<std::string>("Type", "Not Defined");
    auto tLowerType = Plato::tolower(tType);

    if(tLowerType == "scalar function")
    {
        auto tCriterion =
            std::make_shared<Plato::Fluids::ScalarFunction<PhysicsT>>
                (aModel, aDataMap, aInputs, aTag);
        return tCriterion;
    }
    else if(tLowerType == "weighted sum")
    {
        auto tCriterion =
            std::make_shared<Plato::Fluids::WeightedScalarFunction<PhysicsT>>
                (aModel, aDataMap, aInputs, aTag);
        return tCriterion;
    }
    else if(tLowerType == "least squares")
    {
        auto tCriterion =
            std::make_shared<Plato::Fluids::LeastSquaresScalarFunction<PhysicsT>>
                (aModel, aDataMap, aInputs, aTag);
        return tCriterion;
    }
    else
    {
        ANALYZE_THROWERR(std::string("Scalar function in block '") + aTag + "' with Type '" + tType + "' is not supported.")
    }
}

}
// namespace Fluids

}
// namespace Plato
