#pragma once

#include "ScalarFunctionBase.hpp"
#include "problem/parabolic/PhysicsScalarFunction.hpp"
#include "problem/parabolic/TimeIntegratedStateAverageFunction.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace Plato
{

namespace Parabolic
{
/******************************************************************************/
/**
 * \brief Create method
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aDataMap Plato Analyze data map
 * \param [in] aProblemParams parameter input
 * \param [in] aFunctionName name of function in parameter list
 **********************************************************************************/
template <typename PhysicsT>
std::shared_ptr<Plato::Parabolic::ScalarFunctionBase> ScalarFunctionBaseFactory<PhysicsT>::create(
    plato::domain::SpatialModel& aSpatialModel,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aProblemParams,
    const std::string& aFunctionName) const
{
    auto tProblemFunction = aProblemParams.sublist("Criteria").sublist(aFunctionName);
    auto tFunctionType = tProblemFunction.get<std::string>("Type", "Not Defined");

    if (tFunctionType == Plato::Parabolic::physics_scalar_function_name())
    {
        return std::make_shared<Plato::Parabolic::PhysicsScalarFunction<PhysicsT>>(aSpatialModel, aDataMap,
                                                                                   aProblemParams, aFunctionName);
    }
    else if (tFunctionType == plato::parabolic::time_integrated_state_average_function_name())
    {
        return std::make_shared<plato::parabolic::TimeIntegratedStateAverageFunction<PhysicsT>>(
            aSpatialModel, aDataMap, aProblemParams, aFunctionName);
    }
    else
    {
        const std::string tErrorString = std::string("Unknown function Type '") + tFunctionType +
                                         "' specified in function name " + aFunctionName + " ParameterList";
        ANALYZE_THROWERR(tErrorString)
    }
}
}  // namespace Parabolic

}  // namespace Plato
