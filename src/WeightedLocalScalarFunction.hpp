/*
 * WeightedLocalScalarFunction.hpp
 *
 *  Created on: Mar 8, 2020
 */

#pragma once

#include "BLAS2.hpp"
#include "WorksetBase.hpp"
#include "SimplexFadTypes.hpp"
#include "LocalScalarFunctionInc.hpp"
#include "InfinitesimalStrainPlasticity.hpp"
#include "InfinitesimalStrainThermoPlasticity.hpp"
#include "AbstractLocalScalarFunctionInc.hpp"
#include "PathDependentScalarFunctionFactory.hpp"

namespace Plato
{

template<typename PhysicsT>
class WeightedLocalScalarFunction : public Plato::LocalScalarFunctionInc
{
private:
    static constexpr auto mNumSpatialDims = PhysicsT::SimplexT::mNumSpatialDims;           /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::SimplexT::mNumNodesPerCell;         /*!< number of nodes per element */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;      /*!< number of configuration degrees of freedom per element */
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::SimplexT::mNumDofsPerCell;     /*!< number of global degrees of freedom per element */
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::SimplexT::mNumLocalDofsPerCell; /*!< number of local degrees of freedom per element */

    std::string mWeigthedSumFunctionName;        /*!< User defined function name */

    const Plato::SpatialModel & mSpatialModel;
    Plato::DataMap& mDataMap;                    /*!< output database */
    Plato::WorksetBase<PhysicsT> mWorksetBase;   /*!< Assembly routine interface */

    bool mWriteDiagnostics;                      /*!< write diagnostics to console */
    std::vector<std::string> mFunctionNames;     /*!< Vector of function names */
    std::vector<Plato::Scalar> mFunctionWeights; /*!< Vector of function weights */

    std::vector<std::shared_ptr<Plato::LocalScalarFunctionInc>> mLocalScalarFunctionContainer; /*!< Vector of LocalScalarFunctionInc objects */

private:
    /******************************************************************************//**
     * \brief Initialization of Weighted Sum of Local Scalar Functions
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void
    initialize (
        Teuchos::ParameterList & aInputParams)
    {
        if(aInputParams.sublist("Criteria").isSublist(mWeigthedSumFunctionName) == false)
        {
            const auto tError = std::string("UNKNOWN USER DEFINED SCALAR FUNCTION SUBLIST '")
                    + mWeigthedSumFunctionName + "'. USER DEFINED SCALAR FUNCTION SUBLIST '" + mWeigthedSumFunctionName
                    + "' IS NOT DEFINED IN THE INPUT FILE.";
            ANALYZE_THROWERR(tError)
        }

        mFunctionNames.clear();
        mFunctionWeights.clear();
        mLocalScalarFunctionContainer.clear();

        auto tProblemFunctionParams = aInputParams.sublist("Criteria").sublist(mWeigthedSumFunctionName);
        mWriteDiagnostics = tProblemFunctionParams.get<bool>("Write Diagnostics", false);

        if(tProblemFunctionParams.isParameter("Functions") == false)
        {
            const auto tErrorString = std::string("WeightedLocalScalarFunction: 'Functions' Keyword is not defined. ") +
                + "Used the 'Functions' keyword to define each weighted function.";
            ANALYZE_THROWERR(tErrorString)
        }
        auto tFunctionNamesTeuchos = tProblemFunctionParams.get<Teuchos::Array<std::string>>("Functions");
        auto tFunctionNames = tFunctionNamesTeuchos.toVector();

        if(tProblemFunctionParams.isParameter("Weights") == false)
        {
            const auto tErrorString = std::string("WeightedLocalScalarFunction: 'Weights' Keyword is not defined. ") +
                + "Used the 'Weights' keyword to define the weight of each weighted function.";
            ANALYZE_THROWERR(tErrorString)
        }
        auto tFunctionWeightsTeuchos = tProblemFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Weights");
        auto tFunctionWeights = tFunctionWeightsTeuchos.toVector();

        if (tFunctionNames.size() != tFunctionWeights.size())
        {
            const auto tErrorString = std::string("Number of 'Functions' in '") + mWeigthedSumFunctionName
                + "' parameter list does not match the number of 'Weights'";
            ANALYZE_THROWERR(tErrorString)
        }

        Plato::PathDependentScalarFunctionFactory<PhysicsT> tFactory;
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
        {
            mFunctionNames.push_back(tFunctionNames[tFunctionIndex]);
            mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
            auto tScalarFunction = tFactory.create(mSpatialModel, mDataMap, aInputParams, tFunctionNames[tFunctionIndex]);
            mLocalScalarFunctionContainer.push_back(tScalarFunction);
        }
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor for weighted sum of local scalar functions.
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap      output database
     * \param [in] aInputParams  input parameters database
     * \param [in] aName         user-defined function name
    **********************************************************************************/
    WeightedLocalScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) :
        mWeigthedSumFunctionName(aName),
        mWriteDiagnostics(false),
        mSpatialModel(aSpatialModel),
        mDataMap(aDataMap),
        mWorksetBase(aSpatialModel.Mesh)
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Secondary weight sum function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap  output database
    **********************************************************************************/
    WeightedLocalScalarFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        mWeigthedSumFunctionName("Weighted Sum"),
        mWriteDiagnostics(false),
        mSpatialModel(aSpatialModel),
        mDataMap(aDataMap),
        mWorksetBase(aSpatialModel.Mesh)
    {
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~WeightedLocalScalarFunction(){}

    /******************************************************************************//**
     * \brief Write diagnostics to console (activate)
    **********************************************************************************/
    void writeDiagnostics()
    {
        mWriteDiagnostics = true;
    }

    /******************************************************************************//**
     * \brief Add function name
     * \param [in] aName function name
    **********************************************************************************/
    void appendFunctionName(const std::string & aName)
    {
        mFunctionNames.push_back(aName);
    }

    /******************************************************************************//**
     * \brief Add function weight
     * \param [in] aWeight function weight
    **********************************************************************************/
    void appendFunctionWeight(const Plato::Scalar & aWeight)
    {
        mFunctionWeights.push_back(aWeight);
    }

    /******************************************************************************//**
     * \brief Add local scalar function
     * \param [in] aInput scalar function
    **********************************************************************************/
    void appendScalarFunctionBase(const std::shared_ptr<Plato::LocalScalarFunctionInc>& aInput)
    {
        mLocalScalarFunctionContainer.push_back(aInput);
    }

    /******************************************************************************//**
     * \brief Return user defined name
     * return user defined name
    **********************************************************************************/
    std::string name() const override
    {
        return (mWeigthedSumFunctionName);
    }

    /******************************************************************************//**
     * \brief Update physics-based data in between optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
     * \param [in] aTimeStep    current time step
    **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarMultiVector & aGlobalStates,
        const Plato::ScalarMultiVector & aLocalStates,
        const Plato::ScalarVector      & aControls,
        const Plato::TimeData          & aTimeData
    ) const override
    {
        if(mLocalScalarFunctionContainer.empty())
        {
            ANALYZE_THROWERR("LOCAL SCALAR FUNCTION CONTAINER IS EMPTY")
        }

        for(Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mLocalScalarFunctionContainer.size(); tFunctionIndex++)
        {
            mLocalScalarFunctionContainer[tFunctionIndex]->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeData);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate weighted sum of local scalar functions
     *
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             time data object
     *
     * \return weighted sum
    *******************************************************************************/
    Plato::Scalar
    value(
        const Plato::ScalarVector & aCurrentGlobalState,
        const Plato::ScalarVector & aPreviousGlobalState,
        const Plato::ScalarVector & aCurrentLocalState,
        const Plato::ScalarVector & aPreviousLocalState,
        const Plato::ScalarVector & aControls,
        const Plato::TimeData     & aTimeData
    ) const override
    {
        if(mLocalScalarFunctionContainer.empty())
        {
            ANALYZE_THROWERR("LOCAL SCALAR FUNCTION CONTAINER IS EMPTY")
        }

        if(mLocalScalarFunctionContainer.size() != mFunctionWeights.size())
        {
            ANALYZE_THROWERR("DIMENSION MISMATCH: NUMBER OF LOCAL SCALAR FUNCTIONS DOES NOT MATCH THE NUMBER OF WEIGHTS")
        }

        if(mWriteDiagnostics)
        {
            printf("\nTime Step = %f\n", aTimeData.mCurrentTime);
        }

        Plato::Scalar tResult = 0.0;
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mLocalScalarFunctionContainer.size(); tFunctionIndex++)
        {
            const auto tFunctionWeight = mFunctionWeights[tFunctionIndex];
            const auto tFunctionValue = mLocalScalarFunctionContainer[tFunctionIndex]->value(aCurrentGlobalState, aPreviousGlobalState,
                                                                                             aCurrentLocalState, aPreviousLocalState,
                                                                                             aControls, aTimeData);
            const auto tMyFunctionValue = tFunctionWeight * tFunctionValue;

            const auto tFunctionName = mFunctionNames[tFunctionIndex];
            mDataMap.mScalarValues[tFunctionName] = tMyFunctionValue;
            tResult += tMyFunctionValue;

            if(mWriteDiagnostics)
            {
                printf("Scalar Function Name = %s \t Value = %f\n", tFunctionName.c_str(), tMyFunctionValue);
            }
        }

        if(mWriteDiagnostics)
        {
            printf("Weighted Sum Name = %s \t Value = %f\n", mWeigthedSumFunctionName.c_str(), tResult);
        }

        return tResult;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weighted sum of local scalar functions with
     *        respect to (wrt) control parameters
     *
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             time data object
     *
     * \return 2-D view with the gradient of weighted sum of scalar functions wrt
     * control parameters
    **********************************************************************************/
    Plato::ScalarMultiVector gradient_z(const Plato::ScalarVector &aCurrentGlobalState,
                                        const Plato::ScalarVector &aPreviousGlobalState,
                                        const Plato::ScalarVector &aCurrentLocalState,
                                        const Plato::ScalarVector &aPreviousLocalState,
                                        const Plato::ScalarVector &aControls,
                                        const Plato::TimeData     &aTimeData) const override
    {
        const auto tNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tOutput("gradient control workset", tNumCells, mNumNodesPerCell);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mLocalScalarFunctionContainer.size(); tFunctionIndex++)
        {
            const auto tFunctionWeight = mFunctionWeights[tFunctionIndex];
            auto tFunctionGradZ = mLocalScalarFunctionContainer[tFunctionIndex]->gradient_z(aCurrentGlobalState, aPreviousGlobalState,
                                                                                            aCurrentLocalState, aPreviousLocalState,
                                                                                            aControls, aTimeData);
            Plato::blas2::update(tFunctionWeight, tFunctionGradZ, static_cast<Plato::Scalar>(1.0), tOutput);
        }
        return tOutput;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weighted sum of local scalar functions with
     *        respect to (wrt) configuration parameters
     *
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             time data object
     *
     * \return 2-D view with the gradient of weighted sum of scalar functions wrt
     * configuration parameters
    **********************************************************************************/
    Plato::ScalarMultiVector gradient_x(const Plato::ScalarVector &aCurrentGlobalState,
                                        const Plato::ScalarVector &aPreviousGlobalState,
                                        const Plato::ScalarVector &aCurrentLocalState,
                                        const Plato::ScalarVector &aPreviousLocalState,
                                        const Plato::ScalarVector &aControls,
                                        const Plato::TimeData     &aTimeData) const override
    {
        const auto tNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tOutput("gradient configuration workset", tNumCells, mNumConfigDofsPerCell);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mLocalScalarFunctionContainer.size(); tFunctionIndex++)
        {
            const auto tFunctionWeight = mFunctionWeights[tFunctionIndex];
            auto tFunctionGradX = mLocalScalarFunctionContainer[tFunctionIndex]->gradient_x(aCurrentGlobalState, aPreviousGlobalState,
                                                                                            aCurrentLocalState, aPreviousLocalState,
                                                                                            aControls, aTimeData);
            Plato::blas2::update(tFunctionWeight, tFunctionGradX, static_cast<Plato::Scalar>(1.0), tOutput);
        }
        return tOutput;
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt current global states
     *
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             time data object
     *
     * \return workset with partial derivative wrt current global states
    *******************************************************************************/
    Plato::ScalarMultiVector gradient_u(const Plato::ScalarVector & aCurrentGlobalState,
                                        const Plato::ScalarVector & aPreviousGlobalState,
                                        const Plato::ScalarVector & aCurrentLocalState,
                                        const Plato::ScalarVector & aPreviousLocalState,
                                        const Plato::ScalarVector & aControls,
                                        const Plato::TimeData     & aTimeData) const override
    {
        const auto tNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tOutput("gradient current global states workset", tNumCells, mNumGlobalDofsPerCell);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mLocalScalarFunctionContainer.size(); tFunctionIndex++)
        {
            const auto tFunctionWeight = mFunctionWeights[tFunctionIndex];
            auto tFunctionGradCurrentGlobalState =
                mLocalScalarFunctionContainer[tFunctionIndex]->gradient_u(aCurrentGlobalState, aPreviousGlobalState,
                                                                          aCurrentLocalState, aPreviousLocalState,
                                                                          aControls, aTimeData);
            Plato::blas2::update(tFunctionWeight, tFunctionGradCurrentGlobalState, static_cast<Plato::Scalar>(1.0), tOutput);
        }
        return tOutput;
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt previous global states
     *
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             time data object
     *
     * \return workset with partial derivative wrt previous global states
    *******************************************************************************/
    Plato::ScalarMultiVector gradient_up(const Plato::ScalarVector & aCurrentGlobalState,
                                         const Plato::ScalarVector & aPreviousGlobalState,
                                         const Plato::ScalarVector & aCurrentLocalState,
                                         const Plato::ScalarVector & aPreviousLocalState,
                                         const Plato::ScalarVector & aControls,
                                         const Plato::TimeData     & aTimeData) const override
    {
        const auto tNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tOutput("gradient previous global states workset", tNumCells, mNumGlobalDofsPerCell);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mLocalScalarFunctionContainer.size(); tFunctionIndex++)
        {
            const auto tFunctionWeight = mFunctionWeights[tFunctionIndex];
            auto tFunctionGradPreviousGlobalState =
                mLocalScalarFunctionContainer[tFunctionIndex]->gradient_up(aCurrentGlobalState, aPreviousGlobalState,
                                                                           aCurrentLocalState, aPreviousLocalState,
                                                                           aControls, aTimeData);
            Plato::blas2::update(tFunctionWeight, tFunctionGradPreviousGlobalState, static_cast<Plato::Scalar>(1.0), tOutput);
        }
        return tOutput;
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt current local states
     *
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             time data object
     *
     * \return workset with partial derivative wrt current local states
    *******************************************************************************/
    Plato::ScalarMultiVector gradient_c(const Plato::ScalarVector & aCurrentGlobalState,
                                        const Plato::ScalarVector & aPreviousGlobalState,
                                        const Plato::ScalarVector & aCurrentLocalState,
                                        const Plato::ScalarVector & aPreviousLocalState,
                                        const Plato::ScalarVector & aControls,
                                        const Plato::TimeData     & aTimeData) const override
    {
        const auto tNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tOutput("gradient current local states workset", tNumCells, mNumLocalDofsPerCell);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mLocalScalarFunctionContainer.size(); tFunctionIndex++)
        {
            const auto tFunctionWeight = mFunctionWeights[tFunctionIndex];
            auto tFunctionGradCurrentLocalState =
                mLocalScalarFunctionContainer[tFunctionIndex]->gradient_c(aCurrentGlobalState, aPreviousGlobalState,
                                                                          aCurrentLocalState, aPreviousLocalState,
                                                                          aControls, aTimeData);
            Plato::blas2::update(tFunctionWeight, tFunctionGradCurrentLocalState, static_cast<Plato::Scalar>(1.0), tOutput);
        }
        return tOutput;
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt previous local states
     *
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             time data object
     *
     * \return workset with partial derivative wrt previous local states
    *******************************************************************************/
    Plato::ScalarMultiVector gradient_cp(const Plato::ScalarVector & aCurrentGlobalState,
                                         const Plato::ScalarVector & aPreviousGlobalState,
                                         const Plato::ScalarVector & aCurrentLocalState,
                                         const Plato::ScalarVector & aPreviousLocalState,
                                         const Plato::ScalarVector & aControls,
                                         const Plato::TimeData     & aTimeData) const override
    {
        const auto tNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tOutput("gradient previous local states workset", tNumCells, mNumLocalDofsPerCell);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mLocalScalarFunctionContainer.size(); tFunctionIndex++)
        {
            const auto tFunctionWeight = mFunctionWeights[tFunctionIndex];
            auto tFunctionGradPreviousLocalState =
                mLocalScalarFunctionContainer[tFunctionIndex]->gradient_cp(aCurrentGlobalState, aPreviousGlobalState,
                                                                           aCurrentLocalState, aPreviousLocalState,
                                                                           aControls, aTimeData);
            Plato::blas2::update(tFunctionWeight, tFunctionGradPreviousLocalState, static_cast<Plato::Scalar>(1.0), tOutput);
        }
        return tOutput;
    }
};
// class WeightedLocalScalarFunction

}
// namespace Plato


#ifdef PLATOANALYZE_2D
extern template class Plato::WeightedLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<2>>;
extern template class Plato::WeightedLocalScalarFunction<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::WeightedLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<3>>;
extern template class Plato::WeightedLocalScalarFunction<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif
