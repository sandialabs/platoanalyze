/*
 * FluidsFunctionFactory.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "hyperbolic/fluids/IncompressibleFluids.hpp"

#include "hyperbolic/fluids/CriterionVolume.hpp"
#include "hyperbolic/fluids/CriterionFlowRate.hpp"
#include "hyperbolic/fluids/CriterionMeanTemperature.hpp"
#include "hyperbolic/fluids/CriterionThermalCompliance.hpp"
#include "hyperbolic/fluids/CriterionSurfaceThermalFlux.hpp"
#include "hyperbolic/fluids/CriterionMeanSurfacePressure.hpp"
#include "hyperbolic/fluids/CriterionMeanSurfaceTemperature.hpp"

#include "hyperbolic/fluids/PressureResidual.hpp"
#include "hyperbolic/fluids/TemperatureResidual.hpp"
#include "hyperbolic/fluids/VelocityCorrectorResidual.hpp"
#include "hyperbolic/fluids/VelocityPredictorResidual.hpp"

namespace Plato
{

namespace Fluids
{

/**************************************************************************//**
* \struct Vector and scalar function factory.
*
* \brief Responsible for the construction of vector and scalar functions.
******************************************************************************/
struct FunctionFactory
{
public:
    /**************************************************************************//**
    * \fn shared_ptr<AbstractVectorFunction> createVectorFunction
    * \tparam PhysicsT    physics type
    * \tparam EvaluationT Forward Automatic Differentiation evaluation type
    *
    * \brief Responsible for the construction of vector functions.
    *
    * \param [in] aTag     vector function tag/type
    * \param [in] aModel   struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap output database
    * \param [in] aInputs  input file metadata
    *
    * \return shared pointer to an abtract vector function
    ******************************************************************************/
    template
    <typename PhysicsT,
     typename EvaluationT>
    std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>>
    createVectorFunction
    (const std::string          & aTag,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        auto tLowerTag = Plato::tolower(aTag);
        if( tLowerTag == "pressure" )
        {
            return ( std::make_shared<Plato::Fluids::PressureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity corrector" )
        {
            return ( std::make_shared<Plato::Fluids::VelocityCorrectorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "temperature" )
        {
            return ( std::make_shared<Plato::Fluids::TemperatureResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else if( tLowerTag == "velocity predictor" )
        {
            return ( std::make_shared<Plato::Fluids::VelocityPredictorResidual<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
        }
        else
        {
            ANALYZE_THROWERR(std::string("Vector function with tag '") + aTag + "' is not supported.")
        }
    }

    /**************************************************************************//**
    * \fn shared_ptr<AbstractScalarFunction> createScalarFunction
    * \tparam PhysicsT    physics type
    * \tparam EvaluationT Forward Automatic Differentiation evaluation type
    *
    * \brief Responsible for the construction of vector functions.
    *
    * \param [in] aTag     vector function tag/type
    * \param [in] aModel   struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap output database
    * \param [in] aInputs  input file metadata
    *
    * \return shared pointer to an abtract scalar function
    ******************************************************************************/
    template
    <typename PhysicsT,
     typename EvaluationT>
    std::shared_ptr<Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>>
    createScalarFunction
    (const std::string          & aTag,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs)
    {
        if( !aInputs.isSublist("Criteria") )
        {
            ANALYZE_THROWERR("'Criteria' block is not defined.")
        }
        auto tCriteriaList = aInputs.sublist("Criteria");
        if( !tCriteriaList.isSublist(aTag) )
        {
            ANALYZE_THROWERR(std::string("Criteria Block with name '") + aTag + "' is not defined.")
        }
        auto tCriterion = tCriteriaList.sublist(aTag);

        if(!tCriterion.isParameter("Scalar Function Type"))
        {
            ANALYZE_THROWERR(std::string("'Scalar Function Type' keyword is not defined in Criterion with name '") + aTag + "'.")
        }

        auto tCriterionTag = tCriterion.get<std::string>("Scalar Function Type", "Not Defined");
        auto tCriterionLowerTag = Plato::tolower(tCriterionTag);
        
        if( tCriterionLowerTag == "volume" )
        {
            return ( std::make_shared<Plato::Fluids::CriterionVolume<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );
        }else
        if( tCriterionLowerTag == "flow rate" )
        {
            return ( std::make_shared<Plato::Fluids::CriterionFlowRate<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );
        }
        else
        if( tCriterionLowerTag == "thermal flux" )
        {
            return ( std::make_shared<Plato::Fluids::CriterionSurfaceThermalFlux<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );           
        }
        else
        if( tCriterionLowerTag == "thermal compliance" )
        {
            return ( std::make_shared<Plato::Fluids::CriterionThermalCompliance<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );           
        }
        else 
        if( tCriterionLowerTag == "mean temperature" )
        {
            return ( std::make_shared<Plato::Fluids::CriterionMeanTemperature<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );
        }
        else 
        if( tCriterionLowerTag == "mean surface pressure" )
        {
            return ( std::make_shared<Plato::Fluids::CriterionMeanSurfacePressure<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );
        }
        else 
        if( tCriterionLowerTag == "mean surface temperature" )
        {
            return ( std::make_shared<Plato::Fluids::CriterionMeanSurfaceTemperature<PhysicsT, EvaluationT>>
                (aTag, aDomain, aDataMap, aInputs) );
        }
        else
        {
            ANALYZE_THROWERR(std::string("'Scalar Function Type' with tag '") + tCriterionTag
                + "' in Criterion Block '" + aTag + "' is not supported.")
        }
    }
};
// struct FunctionFactory

}
// namespace Fluids

}
// namespace Plato
