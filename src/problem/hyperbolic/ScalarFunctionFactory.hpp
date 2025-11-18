#pragma once

#include <Teuchos_ParameterList.hpp>
#include <memory>

#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/hyperbolic/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
/**
 * \brief Scalar function base factory
 **********************************************************************************/
template <typename PhysicsType>
class ScalarFunctionFactory
{
   public:
    /******************************************************************************/
    /**
     * \brief Constructor
     **********************************************************************************/
    ScalarFunctionFactory() {}

    /******************************************************************************/
    /**
     * \brief Create method
     * \param [in] aMesh mesh database
     * \param [in] aDataMap Plato Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::Hyperbolic::ScalarFunctionBase> create(Plato::SpatialModel& aSpatialModel,
                                                                  Plato::DataMap& aDataMap,
                                                                  Teuchos::ParameterList& aInputParams,
                                                                  std::string& aFunctionName);
};  // class ScalarFunctionFactory

}  // namespace Hyperbolic

}  // namespace Plato

#include "element/BaseExpInstMacros.hpp"
#include "problem/hyperbolic/Mechanics.hpp"
PLATO_ELEMENT_DEC(Plato::Hyperbolic::ScalarFunctionFactory, Plato::Hyperbolic::Mechanics)

#ifdef PLATO_MICROMORPHIC
#include "problem/hyperbolic/micromorphic/MicromorphicMechanics.hpp"
PLATO_ELEMENT_DEC(Plato::Hyperbolic::ScalarFunctionFactory, Plato::Hyperbolic::MicromorphicMechanics)
#endif
