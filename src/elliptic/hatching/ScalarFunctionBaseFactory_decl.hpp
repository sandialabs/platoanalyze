#pragma once

#include <Teuchos_ParameterList.hpp>
#include <memory>

#include "PlatoSequence.hpp"
#include "PlatoStaticsTypes.hpp"
#include "SpatialModel.hpp"
#include "elliptic/hatching/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

/******************************************************************************/
/**
 * \brief Scalar function base factory
 **********************************************************************************/
template <typename PhysicsType>
class ScalarFunctionBaseFactory
{
    using ElementType = typename PhysicsType::ElementType;

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
    std::shared_ptr<Plato::Elliptic::Hatching::ScalarFunctionBase> create(Plato::SpatialModel& aSpatialModel,
                                                                          const Plato::Sequence<ElementType>& aSequence,
                                                                          Plato::DataMap& aDataMap,
                                                                          Teuchos::ParameterList& aInputParams,
                                                                          std::string& aFunctionName);
};
// class ScalarFunctionBaseFactory

}  // namespace Hatching

}  // namespace Elliptic

}  // namespace Plato
