#pragma once

#include <string>

#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/geometric/LeastSquaresFunction_decl.hpp"
#include "problem/geometric/ScalarFunctionBaseFactory.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************/
/**
 * \brief Initialization of Least Squares Function
 * \param [in] aProblemParams input parameters database
 **********************************************************************************/
template <typename PhysicsType>
void LeastSquaresFunction<PhysicsType>::initialize(Teuchos::ParameterList& aProblemParams)
{
    Plato::Geometric::ScalarFunctionBaseFactory<PhysicsType> tFactory;

    auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);

    auto tFunctionNamesArray = tFunctionParams.get<Teuchos::Array<std::string>>("Functions");
    auto tFunctionWeightsArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Weights");
    auto tFunctionGoldValuesArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Gold Values");

    auto tFunctionNames = tFunctionNamesArray.toVector();
    auto tFunctionWeights = tFunctionWeightsArray.toVector();
    auto tFunctionGoldValues = tFunctionGoldValuesArray.toVector();

    if (tFunctionNames.size() != tFunctionWeights.size())
    {
        const std::string tErrorString = std::string("Number of 'Functions' in '") + mFunctionName +
                                         "' parameter list does not equal the number of 'Weights'";
        ANALYZE_THROWERR(tErrorString)
    }

    if (tFunctionNames.size() != tFunctionGoldValues.size())
    {
        const std::string tErrorString = std::string("Number of 'Gold Values' in '") + mFunctionName +
                                         "' parameter list does not equal the number of 'Functions'";
        ANALYZE_THROWERR(tErrorString)
    }

    constexpr Plato::Scalar tDefaultFunctionNormalization{1.0};
    for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
    {
        mFunctions.try_emplace(
            tFunctionNames[tFunctionIndex], tFunctionWeights[tFunctionIndex], tFunctionGoldValues[tFunctionIndex],
            tDefaultFunctionNormalization,
            tFactory.create(mSpatialModel, mDataMap, aProblemParams, tFunctionNames[tFunctionIndex]));
    }
}

/******************************************************************************/
/**
 * \brief Primary least squares function constructor
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aDataMap Plato Analyze data map
 * \param [in] aProblemParams input parameters database
 * \param [in] aName user defined function name
 **********************************************************************************/
template <typename PhysicsType>
LeastSquaresFunction<PhysicsType>::LeastSquaresFunction(const plato::domain::SpatialModel& aSpatialModel,
                                                        Plato::DataMap& aDataMap,
                                                        Teuchos::ParameterList& aProblemParams,
                                                        const std::string& aName)
    : Plato::WorksetBase<ElementType>(aSpatialModel.mMesh),
      mSpatialModel(aSpatialModel),
      mDataMap(aDataMap),
      mFunctionName(aName)
{
    initialize(aProblemParams);
}

/******************************************************************************/
/**
 * \brief Secondary least squares function constructor, used for unit testing / mass properties
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aDataMap Plato Analyze data map
 **********************************************************************************/
template <typename PhysicsType>
LeastSquaresFunction<PhysicsType>::LeastSquaresFunction(const plato::domain::SpatialModel& aSpatialModel,
                                                        Plato::DataMap& aDataMap,
                                                        const unsigned int aPower)
    : Plato::WorksetBase<ElementType>(aSpatialModel.mMesh),
      mSpatialModel(aSpatialModel),
      mDataMap(aDataMap),
      mFunctionName("Least Squares"),
      mPower(aPower)
{
}

/******************************************************************************/
/**
 * \brief Update physics-based parameters within optimization iterations
 * \param [in] aControl 1D view of control variables
 **********************************************************************************/
template <typename PhysicsType>
void LeastSquaresFunction<PhysicsType>::updateProblem(const Plato::ScalarVector& aControl) const
{
    for (const auto& [tName, tFunctionData] : mFunctions)
    {
        tFunctionData.mScalarFunction->updateProblem(aControl);
    }
}

template <typename PhysicsType>
void LeastSquaresFunction<PhysicsType>::appendScalarFunctions(
    std::unordered_map<std::string, LeastSquaresFunctionData> aFunctionMap)
{
    mFunctions = std::move(aFunctionMap);
}

/******************************************************************************/
/**
 * \brief Evaluate least squares function
 * \param [in] aControl 1D view of control variables
 * \return scalar function evaluation
 **********************************************************************************/
template <typename PhysicsType>
Plato::Scalar LeastSquaresFunction<PhysicsType>::value(const Plato::ScalarVector& aControl) const
{
    Plato::Scalar tResult = 0.0;
    for (const auto& [tName, tFunctionData] : mFunctions)
    {
        const Plato::Scalar tFunctionWeight = tFunctionData.mWeight;
        const Plato::Scalar tFunctionGoldValue = tFunctionData.mGoldValue;
        const Plato::Scalar tFunctionScale = tFunctionData.mNormalization;
        const Plato::Scalar tFunctionValue = tFunctionData.mScalarFunction->value(aControl);
        tResult += tFunctionWeight * std::pow((tFunctionValue - tFunctionGoldValue) / tFunctionScale, mPower);

        const Plato::Scalar tPercentDiff = std::abs(tFunctionGoldValue) > 0.0
                                               ? 100.0 * (tFunctionValue - tFunctionGoldValue) / tFunctionGoldValue
                                               : (tFunctionValue - tFunctionGoldValue);
        std::cout << std::format(
            "{:.20s} = {:12.4e} * (({:12.4e} - {:12.4e}) / {:12.4e})^{} =  {:12.4e} (PercDiff = {:10.1f})\n",
            tName.c_str(), tFunctionWeight, tFunctionValue, tFunctionGoldValue, tFunctionScale, mPower,
            tFunctionWeight * std::pow((tFunctionValue - tFunctionGoldValue) / tFunctionScale, mPower), tPercentDiff);
    }
    return tResult;
}

/******************************************************************************/
/**
 * \brief Evaluate gradient of the least squares function with respect to (wrt) the configuration parameters
 * \param [in] aControl 1D view of control variables
 * \return 1D view with the gradient of the scalar function wrt the configuration parameters
 **********************************************************************************/
template <typename PhysicsType>
Plato::ScalarVector LeastSquaresFunction<PhysicsType>::gradient_x(const Plato::ScalarVector& aControl) const
{
    const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
    Plato::ScalarVector tGradientX("gradient configuration", tNumDofs);
    for (const auto& [tName, tFunctionData] : mFunctions)
    {
        const Plato::Scalar tPower = mPower;
        const Plato::Scalar tFunctionWeight = tFunctionData.mWeight;
        const Plato::Scalar tFunctionGoldValue = tFunctionData.mGoldValue;
        const Plato::Scalar tFunctionScale = tFunctionData.mNormalization;
        const Plato::Scalar tFunctionValue = tFunctionData.mScalarFunction->value(aControl);
        const Plato::ScalarVector tFunctionGradX = tFunctionData.mScalarFunction->gradient_x(aControl);
        Kokkos::parallel_for(
            "Least Squares Function Summation Grad X", Kokkos::RangePolicy<>(0, tNumDofs),
            KOKKOS_LAMBDA(const Plato::OrdinalType& tDof) {
                tGradientX(tDof) += tPower * tFunctionWeight *
                                    std::pow(tFunctionValue - tFunctionGoldValue, tPower - 1.0) * tFunctionGradX(tDof) /
                                    (tFunctionScale * tFunctionScale);
            });
    }
    return tGradientX;
}

/******************************************************************************/
/**
 * \brief Evaluate gradient of the least squares function with respect to (wrt) the control variables
 * \param [in] aControl 1D view of control variables
 * \return 1D view with the gradient of the scalar function wrt the control variables
 **********************************************************************************/
template <typename PhysicsType>
Plato::ScalarVector LeastSquaresFunction<PhysicsType>::gradient_z(const Plato::ScalarVector& aControl) const
{
    const Plato::OrdinalType tNumDofs = mNumNodes;
    Plato::ScalarVector tGradientZ("gradient control", tNumDofs);
    for (const auto& [tName, tFunctionData] : mFunctions)
    {
        const Plato::Scalar tPower = mPower;
        const Plato::Scalar tFunctionWeight = tFunctionData.mWeight;
        const Plato::Scalar tFunctionGoldValue = tFunctionData.mGoldValue;
        const Plato::Scalar tFunctionScale = tFunctionData.mNormalization;
        const Plato::Scalar tFunctionValue = tFunctionData.mScalarFunction->value(aControl);
        const Plato::ScalarVector tFunctionGradZ = tFunctionData.mScalarFunction->gradient_z(aControl);
        Kokkos::parallel_for(
            "Least Squares Function Summation Grad Z", Kokkos::RangePolicy<>(0, tNumDofs),
            KOKKOS_LAMBDA(const Plato::OrdinalType& tDof) {
                tGradientZ(tDof) += tPower * tFunctionWeight *
                                    std::pow(tFunctionValue - tFunctionGoldValue, tPower - 1) * tFunctionGradZ(tDof) /
                                    (tFunctionScale * tFunctionScale);
            });
    }
    return tGradientZ;
}
}  // namespace Geometric

}  // namespace Plato
