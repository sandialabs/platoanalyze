#pragma once

#include "local_operations/optimization/Heaviside.hpp"
#include "local_operations/optimization/MakeFunctions.hpp"
#include "local_operations/optimization/NoPenalty.hpp"
#include "local_operations/optimization/Ramp.hpp"
#include "local_operations/optimization/Simp.hpp"
#include "problem/geometric/AbstractScalarFunction.hpp"
#include "problem/geometric/GeometryMisfit.hpp"
#include "problem/geometric/Volume.hpp"
#include "utilities/PlatoUtilities.hpp"

namespace Plato
{

namespace GeometryFactory
{
/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Geometric::AbstractScalarFunction<EvaluationType>> createScalarFunction(
        const plato::domain::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aParamList,
        std::string aFuncType,
        const std::string& aFuncName)
    {
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if (tLowerFuncType == "volume")
        {
            return Plato::makeScalarFunction<EvaluationType, Plato::Geometric::Volume>(aSpatialDomain, aDataMap,
                                                                                       aParamList, aFuncName);
        }
        else if (tLowerFuncType == "geometry misfit")
        {
            return std::make_shared<Plato::Geometric::GeometryMisfit<EvaluationType>>(aSpatialDomain, aDataMap,
                                                                                      aParamList, aFuncName);
        }
        else
        {
            ANALYZE_THROWERR(std::string("Unknown 'Objective' of type '") + tLowerFuncType +
                             "' specified in 'Plato Problem' ParameterList");
        }
    }
};

}  // namespace GeometryFactory

}  // namespace Plato

#include "problem/geometric/GeometricalElement.hpp"

namespace Plato
{
template <typename TopoElementType>
class Geometrical
{
   public:
    typedef Plato::GeometryFactory::FunctionFactory FunctionFactory;
    using ElementType = GeometricalElement<TopoElementType>;
};
// class Geometrical

}  // namespace Plato
