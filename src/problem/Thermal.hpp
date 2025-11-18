#ifndef PLATO_THERMAL_HPP
#define PLATO_THERMAL_HPP

#include "problem/parabolic/AbstractScalarFunction.hpp"
#include "problem/parabolic/AbstractVectorFunction.hpp"

#ifdef PLATO_PARABOLIC
#include "problem/parabolic/HeatEquationResidual.hpp"
#include "problem/parabolic/InternalThermalEnergy.hpp"
#include "problem/parabolic/TemperatureAverage.hpp"
#endif

#include "local_operations/optimization/MakeFunctions.hpp"
#include "problem/elliptic/AbstractScalarFunction.hpp"
#include "problem/elliptic/AbstractVectorFunction.hpp"
#include "problem/elliptic/FluxPNorm.hpp"
#include "problem/elliptic/InternalThermalEnergy.hpp"
#include "problem/elliptic/ThermostaticResidual.hpp"

namespace Plato
{

namespace ThermalFactory
{
/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>> createVectorFunction(
        const Plato::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aProblemParams,
        std::string aPDE)
    {
        auto tLowerPDE = Plato::tolower(aPDE);
        if (tLowerPDE == "elliptic")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Elliptic::ThermostaticResidual>(
                aSpatialDomain, aDataMap, aProblemParams, aPDE);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>> createVectorFunctionParabolic(
        const Plato::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aProblemParams,
        std::string aPDE)
    {
#ifdef PLATO_PARABOLIC
        auto tLowerPDE = Plato::tolower(aPDE);
        if (tLowerPDE == "parabolic")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Parabolic::HeatEquationResidual>(
                aSpatialDomain, aDataMap, aProblemParams, aPDE);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
#else
        ANALYZE_THROWERR("Plato Analyze was not compiled with parabolic physics.");
#endif
    }

    template <typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> createScalarFunction(
        const Plato::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aProblemParams,
        std::string aFuncType,
        std::string aFuncName)
    {
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if (tLowerFuncType == "internal thermal energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::InternalThermalEnergy>(
                aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if (tLowerFuncType == "flux p-norm")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Elliptic::FluxPNorm>(aSpatialDomain, aDataMap,
                                                                                         aProblemParams, aFuncName);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }

    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>> createScalarFunctionParabolic(
        const Plato::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aProblemParams,
        std::string aFuncType,
        std::string aFuncName)
    {
#ifdef PLATO_PARABOLIC
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if (tLowerFuncType == "internal thermal energy")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Parabolic::InternalThermalEnergy>(
                aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else if (tLowerFuncType == "temperature average")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Parabolic::TemperatureAverage>(
                aSpatialDomain, aDataMap, aProblemParams, aFuncName);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
#else
        ANALYZE_THROWERR("Plato Analyze was not compiled with parabolic physics.");
#endif
    }
};

}  // namespace ThermalFactory

}  // namespace Plato

#include "element/ThermalElement.hpp"

namespace Plato
{
/******************************************************************************/
/**
 * \brief Concrete class for use as the Physics template argument in
 *        Plato::Elliptic::Problem
 **********************************************************************************/
template <typename TopoElementType>
class Thermal
{
   public:
    typedef Plato::ThermalFactory::FunctionFactory FunctionFactory;
    using ElementType = ThermalElement<TopoElementType>;
};
}  // namespace Plato

#endif
