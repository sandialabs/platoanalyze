#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "domain/WorksetBase.hpp"
#include "problem/geometric/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Geometric
{
struct LeastSquaresFunctionData
{
    Plato::Scalar mWeight;
    Plato::Scalar mGoldValue;
    Plato::Scalar mNormalization;
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> mScalarFunction;
};

/******************************************************************************/
/**
 * \brief Least Squares function class \f$ F(x) = \sum_{i = 1}^{n} w_i * (f_i(x) - gold_i(x))^2 \f$
 **********************************************************************************/
template <typename PhysicsType>
class LeastSquaresFunction : public Plato::Geometric::ScalarFunctionBase,
                             public Plato::WorksetBase<typename PhysicsType::ElementType>
{
   public:
    /******************************************************************************/
    /**
     * \brief Primary least squares function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
     **********************************************************************************/
    LeastSquaresFunction(const plato::domain::SpatialModel& aSpatialModel,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aProblemParams,
                         const std::string& aName);

    /******************************************************************************/
    /**
     * \brief Secondary least squares function constructor, used for unit testing / mass properties
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     **********************************************************************************/
    LeastSquaresFunction(const plato::domain::SpatialModel& aSpatialModel,
                         Plato::DataMap& aDataMap,
                         const unsigned int aPower);

    /// @brief append an existing map @a aFunctionMap from function name to a struct containing the scalar function data for use in least squares computation
    void appendScalarFunctions(std::unordered_map<std::string, LeastSquaresFunctionData> aFunctionMap);

    /******************************************************************************/
    /**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector& aControl) const override;

    /******************************************************************************/
    /**
     * \brief Evaluate least squares function
     * \param [in] aControl 1D view of control variables
     * \return scalar function evaluation
     **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector& aControl) const override;

    /******************************************************************************/
    /**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
     **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector& aControl) const override;

    /******************************************************************************/
    /**
     * \brief Evaluate gradient of the least squares function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the control variables
     **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector& aControl) const override;

   private:
    /// @brief Initialize from parameter list
    void initialize(Teuchos::ParameterList& aProblemParams);

   private:
    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::WorksetBase<ElementType>::mNumNodes;

    const plato::domain::SpatialModel& mSpatialModel;

    Plato::DataMap& mDataMap;

    std::string mFunctionName;

    unsigned int mPower = 2;

    std::unordered_map<std::string, LeastSquaresFunctionData> mFunctions;
};
// class LeastSquaresFunction

}  // namespace Geometric

}  // namespace Plato
