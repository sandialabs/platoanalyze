#pragma once

#include <memory>

#include "problem/elliptic/AbstractScalarFunction.hpp"
#include "problem/elliptic/stabilized/AbstractVectorFunction.hpp"
#include "problem/elliptic/stabilized/PressureGradientProjectionResidual.hpp"
#include "problem/elliptic/stabilized/Projection.hpp"
#include "problem/elliptic/stabilized/ThermoelastostaticResidual.hpp"
#include "problem/parabolic/AbstractScalarFunction.hpp"
#include "problem/parabolic/AbstractVectorFunction.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace Plato
{

namespace Stabilized
{

namespace ThermomechanicsFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Stabilized::AbstractVectorFunction<EvaluationType>> createVectorFunction(
        const Plato::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aParamList,
        const std::string& aPDE)
    /******************************************************************************/
    {
        auto tLowerPDE = Plato::tolower(aPDE);

        if (tLowerPDE == "elliptic")
        {
            return Plato::makeVectorFunction<EvaluationType, Plato::Stabilized::ThermoelastostaticResidual>(
                aSpatialDomain, aDataMap, aParamList, aPDE);
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }

    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>> createVectorFunctionParabolic(
        const Plato::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aParamList,
        const std::string& strVectorFunctionType)
    /******************************************************************************/
    {
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>> createScalarFunction(
        const Plato::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aParamList,
        const std::string& aStrScalarFunctionType,
        const std::string& aStrScalarFunctionName)
    /******************************************************************************/
    {
        {
            ANALYZE_THROWERR("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }
    /******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>> createScalarFunctionParabolic(
        const Plato::SpatialDomain& aSpatialDomain,
        Plato::DataMap& aDataMap,
        Teuchos::ParameterList& aParamList,
        const std::string& strScalarFunctionType,
        const std::string& aStrScalarFunctionName)
    /******************************************************************************/
    {
        {
            ANALYZE_THROWERR("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
};  // struct FunctionFactory

}  // namespace ThermomechanicsFactory
}  // namespace Stabilized
}  // namespace Plato

#include "problem/elliptic/stabilized/ThermomechanicsElement.hpp"

namespace Plato
{
namespace Stabilized
{
/****************************************************************************/
/**
 * \brief Concrete class for use as the PhysicsType template argument in Problem
 *******************************************************************************/
template <typename TopoElementType>
class Thermomechanics
{
   public:
    typedef Plato::Stabilized::ThermomechanicsFactory::FunctionFactory FunctionFactory;

    using ElementType = Plato::Stabilized::ThermomechanicsElement<TopoElementType>;
    using ProjectorType = typename Plato::Stabilized::Projection<TopoElementType,
                                                                 ElementType::mNumDofsPerNode,
                                                                 ElementType::mPressureDofOffset,
                                                                 /* numProjectionDofs=*/1>;
};

}  // namespace Stabilized
}  // namespace Plato
