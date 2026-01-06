#pragma once

#include "problem/elliptic/DivisionFunction.hpp"
#include "problem/elliptic/PhysicsScalarFunction.hpp"
#include "problem/elliptic/ScalarFunctionBase.hpp"
#include "problem/elliptic/SolutionFunction.hpp"
#include "problem/elliptic/VolumeAverageCriterion.hpp"
#include "problem/elliptic/finite_deformation_mechanics/VarianceFunction.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
/**
 * \brief Create method
 * \param [in] aSpatialModel Plato
 *Analyze spatial model \param [in]
 *aDataMap Plato and Analyze data map
 * \param [in] aProblemParams parameter
 *input \param [in] aFunctionName name
 *of function in parameter list
 **********************************************************************************/
template <typename PhysicsType>
std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> ScalarFunctionBaseFactory<PhysicsType>::create(
    const plato::domain::SpatialModel& aSpatialModel,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aProblemParams,
    const std::string& aFunctionName) const
{
    auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
    auto tFunctionType = tFunctionParams.get<std::string>("Type", "Not Defined");

    if (tFunctionType == "Solution")
    {
        return std::make_shared<SolutionFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
    }
    else if (tFunctionType == "Division")
    {
        return std::make_shared<DivisionFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
    }
    else if (tFunctionType == "Volume Average Criterion")
    {
        return std::make_shared<VolumeAverageCriterion<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams,
                                                                     aFunctionName);
    }
    else if (tFunctionType == "Scalar Function")
    {
        return std::make_shared<PhysicsScalarFunction<PhysicsType>>(aSpatialModel, aDataMap, aProblemParams,
                                                                    aFunctionName);
    }
    else if (tFunctionType == "Variance Function")
    {
        return std::make_shared<plato::elliptic::finite_deformation_mechanics::VarianceFunction<PhysicsType>>(
            aSpatialModel, aDataMap, aProblemParams, aFunctionName);
    }
    else
    {
        return nullptr;
    }
    return nullptr;
}

}  // namespace Elliptic

}  // namespace Plato
