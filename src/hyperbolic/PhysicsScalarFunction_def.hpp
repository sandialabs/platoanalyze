#pragma once

#include "hyperbolic/PhysicsScalarFunction_decl.hpp"

namespace Plato
{

namespace Hyperbolic
{
    /******************************************************************************//**
     * \brief Initialization of Hyperbolic Physics Scalar Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    initialize(Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsType::FunctionFactory tFactory;

        auto tProblemDefault = aInputParams.sublist("Criteria").sublist(mFunctionName);
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", "");

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mValueFunctions[tName] =
                tFactory.template createScalarFunction<Residual>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientUFunctions[tName] =
                tFactory.template createScalarFunction<GradientU>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientVFunctions[tName] =
                tFactory.template createScalarFunction<GradientV>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientAFunctions[tName] =
                tFactory.template createScalarFunction<GradientA>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientXFunctions[tName] =
                tFactory.template createScalarFunction<GradientX>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientZFunctions[tName] =
                tFactory.template createScalarFunction<GradientZ>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
        }
    }

    /******************************************************************************//**
     * \brief Primary physics scalar function inc constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    PhysicsScalarFunction<PhysicsType>::
    PhysicsScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aName
    ):
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Secondary physics scalar function inc constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    template<typename PhysicsType>
    PhysicsScalarFunction<PhysicsType>::
    PhysicsScalarFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
      Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
      mSpatialModel (aSpatialModel),
      mDataMap      (aDataMap),
      mFunctionName ("Undefined Name")
    {
    }

    /******************************************************************************//**
     * \brief Evaluate physics scalar function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar physics function evaluation
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::Scalar
    PhysicsScalarFunction<PhysicsType>::
    value(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep
    ) const
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using StateScalar       = typename Residual::StateScalarType;
        using StateDotScalar    = typename Residual::StateDotScalarType;
        using StateDotDotScalar = typename Residual::StateDotDotScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        auto tNumSteps = tStates.extent(0);

        std::vector<ResultScalar> tValues(tNumSteps, 0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create result view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);
            mDataMap.scalarVectors[mValueFunctions.at(tName)->getName()] = tResult;

            Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", tNumCells, mNumDofsPerCell);

            for( decltype(tNumSteps) tStepIndex = 1; tStepIndex < tNumSteps; ++tStepIndex )
            {
                // workset state
                //
                auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tState, tStateWS, tDomain);

                // workset state dot
                //
                auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tStateDot, tStateDotWS, tDomain);

                // workset state dot dot
                //
                auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mValueFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

                // sum across elements
                //
                tValues[tStepIndex] += Plato::local_result_sum<Plato::Scalar>( tNumCells, tResult);
            }
        }
        ResultScalar tReturnVal(0.0);
        for( decltype(tNumSteps) tStepIndex = 1; tStepIndex < tNumSteps; ++tStepIndex )
        {
          auto tName = mSpatialModel.Domains.front().getDomainName();
          mValueFunctions.at(tName)->postEvaluate( tValues[tStepIndex] );
          tReturnVal += tValues[tStepIndex];
        }

        return tReturnVal;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_x(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using StateScalar       = typename GradientX::StateScalarType;
        using StateDotScalar    = typename GradientX::StateDotScalarType;
        using StateDotDotScalar = typename GradientX::StateDotDotScalarType;
        using ControlScalar     = typename GradientX::ControlScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create return view
        //
        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims*mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", tNumCells, mNumDofsPerCell);

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            auto tNumSteps = tStates.extent(0);
            auto tLastStepIndex = tNumSteps - 1;
            for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

                // workset state
                //
                auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tState, tStateWS, tDomain);

                // workset state dot
                //
                auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tStateDot, tStateDotWS, tDomain);

                // workset state dot dot
                //
                auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientXFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

                Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                    (tDomain, mConfigEntryOrdinal, tResult, tObjGradientX);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientXFunctions.at(tName)->postEvaluate( tObjGradientX, tValue );

        return tObjGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step
     * \param [in] aStepIndex step index
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_u(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientU::ConfigScalarType;
        using StateScalar       = typename GradientU::StateScalarType;
        using StateDotScalar    = typename GradientU::StateDotScalarType;
        using StateDotDotScalar = typename GradientU::StateDotDotScalarType;
        using ControlScalar     = typename GradientU::ControlScalarType;
        using ResultScalar      = typename GradientU::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", mNumDofsPerNode * mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            // workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", tNumCells, mNumDofsPerCell);
            auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tState, tStateWS, tDomain);

            // workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tStateDot, tStateDotWS, tDomain);

            // workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

            // evaluate function
            //
            mGradientUFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientU);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientUFunctions.at(tName)->postEvaluate( tObjGradientU, tValue );

        return tObjGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state dot variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step
     * \param [in] aStepIndex step index
     * \return 1D view with the gradient of the physics scalar function wrt the state dot variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_v(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientV::ConfigScalarType;
        using StateScalar       = typename GradientV::StateScalarType;
        using StateDotScalar    = typename GradientV::StateDotScalarType;
        using StateDotDotScalar = typename GradientV::StateDotDotScalarType;
        using ControlScalar     = typename GradientV::ControlScalarType;
        using ResultScalar      = typename GradientV::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientV("objective gradient state", mNumDofsPerNode*mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            // workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", tNumCells, mNumDofsPerCell);
            auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tState, tStateWS, tDomain);

            // workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tStateDot, tStateDotWS, tDomain);

            // workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

            // evaluate function
            //
            mGradientVFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientV);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientVFunctions.at(tName)->postEvaluate( tObjGradientV, tValue );

        return tObjGradientV;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state dot dot variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step
     * \param [in] aStepIndex step index
     * \return 1D view with the gradient of the physics scalar function wrt the state dot dot variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_a(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientA::ConfigScalarType;
        using StateScalar       = typename GradientA::StateScalarType;
        using StateDotScalar    = typename GradientA::StateDotScalarType;
        using StateDotDotScalar = typename GradientA::StateDotDotScalarType;
        using ControlScalar     = typename GradientA::ControlScalarType;
        using ResultScalar      = typename GradientA::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientA("objective gradient state", mNumDofsPerNode * mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            // workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", tNumCells, mNumDofsPerCell);
            auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tState, tStateWS, tDomain);

            // workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tStateDot, tStateDotWS, tDomain);

            // workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<ElementType>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

            // evaluate function
            //
            mGradientAFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientA);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientAFunctions.at(tName)->postEvaluate( tObjGradientA, tValue );

        return tObjGradientA;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    template<typename PhysicsType>
    Plato::ScalarVector
    PhysicsScalarFunction<PhysicsType>::
    gradient_z(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep
    ) const
    {
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using StateScalar       = typename GradientZ::StateScalarType;
        using StateDotScalar    = typename GradientZ::StateDotScalarType;
        using StateDotDotScalar = typename GradientZ::StateDotDotScalarType;
        using ControlScalar     = typename GradientZ::ControlScalarType;
        using ResultScalar      = typename GradientZ::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        auto tNumSteps = tStates.extent(0);

        // create return vector
        //
        Plato::ScalarMultiVector tObjGradientZSteps("objective gradient wrt control", tNumSteps, mNumNodes);

        std::vector<Plato::Scalar> tValues(tNumSteps, 0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", tNumCells, mNumDofsPerCell);

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

            // create result view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            for( decltype(tNumSteps) tStepIndex = 1; tStepIndex < tNumSteps; ++tStepIndex )
            {
                // workset state
                //
                auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tState, tStateWS, tDomain);

                // workset state dot
                //
                auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tStateDot, tStateDotWS, tDomain);

                // workset state dot dot
                //
                auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<ElementType>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientZFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

                Plato::ScalarVector tObjGradientZStep = Kokkos::subview(tObjGradientZSteps, tStepIndex, Kokkos::ALL());
                Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>
                    (tDomain, mControlEntryOrdinal, tResult, tObjGradientZStep);

                tValues[tStepIndex] += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        Plato::ScalarVector tObjGradientZ("objective gradient wrt control", mNumNodes);
        for( decltype(tNumSteps) tStepIndex = 1; tStepIndex < tNumSteps; ++tStepIndex )
        {
          Plato::ScalarVector tObjGradientZStep = Kokkos::subview(tObjGradientZSteps, tStepIndex, Kokkos::ALL());
          auto tName = mSpatialModel.Domains.front().getDomainName();
          mGradientZFunctions.at(tName)->postEvaluate( tObjGradientZStep, tValues[tStepIndex] );
          Plato::blas1::axpy(1.0, tObjGradientZStep, tObjGradientZ);
        }

        return tObjGradientZ;
    }

    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    template<typename PhysicsType>
    void
    PhysicsScalarFunction<PhysicsType>::
    setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    template<typename PhysicsType>
    std::string
    PhysicsScalarFunction<PhysicsType>::
    name() const
    {
        return mFunctionName;
    }

} //namespace Hyperbolic

} //namespace Plato
