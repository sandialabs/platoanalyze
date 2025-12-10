#pragma once

#include <Teuchos_ParameterList.hpp>

#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/geometric/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************/
/**
 * \brief Scalar function base factory
 **********************************************************************************/
template <typename PhysicsT>
class ScalarFunctionBaseFactory
{
   public:
    /******************************************************************************/
    /**
     * \brief Constructor
     **********************************************************************************/
    ScalarFunctionBaseFactory() {}

    /******************************************************************************/
    /**
     * \brief Create method
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> create(const plato::domain::SpatialModel& aSpatialModel,
                                                                 Plato::DataMap& aDataMap,
                                                                 Teuchos::ParameterList& aInputParams,
                                                                 std::string& aFunctionName);
};
// class ScalarFunctionBaseFactory

}  // namespace Geometric

}  // namespace Plato
