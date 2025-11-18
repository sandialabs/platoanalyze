#pragma once

#include <Teuchos_ParameterList.hpp>

#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/parabolic/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Parabolic
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
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::Parabolic::ScalarFunctionBase> create(Plato::SpatialModel& aSpatialModel,
                                                                 Plato::DataMap& aDataMap,
                                                                 Teuchos::ParameterList& aInputParams,
                                                                 std::string& aFunctionName);
};
// class ScalarFunctionBaseFactory

}  // namespace Parabolic

}  // namespace Plato
