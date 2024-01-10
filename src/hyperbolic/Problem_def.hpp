#pragma once

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "BLAS1.hpp"

#include "ApplyConstraints.hpp"
#include "EssentialBCs.hpp"
#include "AnalyzeOutput.hpp"
#include "AnalyzeMacros.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "Plato_Solve.hpp"
#include "ComputedField.hpp"

#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/ScalarFunctionFactory.hpp"

namespace Plato
{

namespace Hyperbolic
{
    template<typename PhysicsType>
    Problem<PhysicsType>::
    Problem(
      Plato::Mesh              aMesh,
      Teuchos::ParameterList & aProblemParams,
      Comm::Machine            aMachine
    ) :
        AbstractProblem  (aMesh, aProblemParams),
        mSpatialModel    (aMesh, aProblemParams, mDataMap),
        mPDEConstraint   (mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint")),
        mSaveState       (aProblemParams.sublist("Hyperbolic").isType<Teuchos::Array<std::string>>("Plottable")),
        mInitDisplacement ("Init Displacement", mPDEConstraint.size()),
        mInitVelocity     ("Init Velocity",     mPDEConstraint.size()),
        mInitAcceleration ("Init Acceleration", mPDEConstraint.size()),
        mJacobianU     (Teuchos::null),
        mJacobianV     (Teuchos::null),
        mJacobianA     (Teuchos::null),
        mStateBoundaryConditions      (aProblemParams.sublist("State Essential Boundary Conditions",false), aMesh),
        mStateDotBoundaryConditions   (aProblemParams.sublist("State Dot Essential Boundary Conditions",false), aMesh),
        mStateDotDotBoundaryConditions(aProblemParams.sublist("State Dot Dot Essential Boundary Conditions",false), aMesh),
        mPDE           (aProblemParams.get<std::string>("PDE Constraint")),
        mPhysics       (aProblemParams.get<std::string>("Physics"))
    {
        parseIntegrator(aProblemParams);

        allocateStateData();

        parseCriteria(aProblemParams);

        parseComputedFields(aProblemParams, aMesh);

        parseInitialState(aProblemParams);

        parseLinearSolver(aProblemParams, aMesh, aMachine);
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    parseIntegrator(
      Teuchos::ParameterList & aProblemParams
    )
    {
        auto tParamsGiven = aProblemParams.isSublist("Time Integration");

        if (!tParamsGiven)
        {
            throw std::runtime_error("Required 'Time Integration' ParameterList is missing.");
        }

        auto tIntegratorParams = aProblemParams.sublist("Time Integration");

        if (tIntegratorParams.isType<bool>("A-Form"))
        {
            auto tAForm = tIntegratorParams.get<bool>("A-Form");
            mUForm = !tAForm;
        }
        else
        {
            mUForm = true;
        }

        auto tMaxEigenvalue = mPDEConstraint.getMaxEigenvalue();

        if (mUForm)
            mIntegrator = std::make_shared<Plato::NewmarkIntegratorUForm>(tIntegratorParams, tMaxEigenvalue);
        else
            mIntegrator = std::make_shared<Plato::NewmarkIntegratorAForm>(tIntegratorParams, tMaxEigenvalue);

        mNumSteps = mIntegrator->getNumSteps();
        mTimeStep = mIntegrator->getTimeStep();
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    allocateStateData()
    {
        mDisplacement = Plato::ScalarMultiVector("Displacement", mNumSteps, mPDEConstraint.size());
        mVelocity     = Plato::ScalarMultiVector("Velocity",     mNumSteps, mPDEConstraint.size());
        mAcceleration = Plato::ScalarMultiVector("Acceleration", mNumSteps, mPDEConstraint.size());
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    parseCriteria(
      Teuchos::ParameterList & aProblemParams
    )
    {
        if(aProblemParams.isSublist("Criteria"))
        {
            Plato::Hyperbolic::ScalarFunctionFactory<PhysicsType> tFunctionBaseFactory;

            auto tCriteriaParams = aProblemParams.sublist("Criteria");
            for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
                std::string tName = tCriteriaParams.name(tIndex);

                TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
                  " Parameter in Criteria block not valid.  Expect lists only.");

                {
                    auto tCriterion = tFunctionBaseFactory.create(mSpatialModel, mDataMap, aProblemParams, tName);
                    if( tCriterion != nullptr )
                    {
                        mCriteria[tName] = tCriterion;
                    }
                }
            }
            if( mCriteria.size() )
            {
                auto tLength = mPDEConstraint.size();
                mAdjoints_U = Plato::ScalarMultiVector("MyAdjoint U", mNumSteps, tLength);
                mAdjoints_V = Plato::ScalarMultiVector("MyAdjoint V", mNumSteps, tLength);
                mAdjoints_A = Plato::ScalarMultiVector("MyAdjoint A", mNumSteps, tLength);
            }
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    parseComputedFields(
      Teuchos::ParameterList & aProblemParams,
      Plato::Mesh              aMesh
    )
    {
        if(aProblemParams.isSublist("Computed Fields"))
        {
          mComputedFields = Teuchos::rcp(new Plato::ComputedFields<ElementType::mNumSpatialDims>(aMesh, aProblemParams.sublist("Computed Fields")));
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    parseInitialState(
      Teuchos::ParameterList & aProblemParams
    )
    {
        if(aProblemParams.isSublist("Initial State"))
        {
            if(mComputedFields == Teuchos::null) {
              ANALYZE_THROWERR("No 'Computed Fields' have been defined");
            }

            auto tDofNames = mPDEConstraint.getDofNames();
            auto tDofDotNames = mPDEConstraint.getDofDotNames();

            auto tInitStateParams = aProblemParams.sublist("Initial State");
            for (auto i = tInitStateParams.begin(); i != tInitStateParams.end(); ++i) {
                const auto &tEntry = tInitStateParams.entry(i);
                const auto &tName  = tInitStateParams.name(i);

                if (tEntry.isList())
                {
                    auto& tStateList = tInitStateParams.sublist(tName);
                    auto tFieldName = tStateList.get<std::string>("Computed Field");
                    int tDofIndex = -1;
                    for (int j = 0; j < tDofNames.size(); ++j)
                    {
                        if (tDofNames[j] == tName) {
                           tDofIndex = j;
                           break;
                        }
                    }
                    if (tDofIndex != -1)
                    {
                        mComputedFields->get(tFieldName, tDofIndex, tDofNames.size(), mInitDisplacement);
                    }
                    else
                    {
                        for (int j = 0; j < tDofDotNames.size(); ++j)
                        {
                            if (tDofDotNames[j] == tName) {
                               tDofIndex = j;
                               break;
                            }
                        }
                        if (tDofIndex != -1)
                        {
                            mComputedFields->get(tFieldName, tDofIndex, tDofDotNames.size(), mInitVelocity);
                        }
                        else
                        {
                            ANALYZE_THROWERR("Attempted to initialize state variable that doesn't exist.");
                        }
                    }
                }
            }
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    parseLinearSolver(
      Teuchos::ParameterList & aProblemParams,
      Plato::Mesh              aMesh,
      Comm::Machine            aMachine
    )
    {
        Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"));
        mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode);
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    output(const std::string& aFilepath)
    {
        auto tDataMap = this->getDataMap();
        auto tSolution = this->getSolution();
        auto tSolutionOutput = mPDEConstraint.getSolutionStateOutputData(tSolution);
        Plato::universal_solution_output(aFilepath, tSolutionOutput, tDataMap, mSpatialModel.Mesh);
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    applyConstraints(
      const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
      const Plato::ScalarVector & aVector
    )
    {
        if (mUForm)
        {
            this->applyConstraintType(aMatrix, aVector, mStateBcDofs, mStateBcValues);
        }
        else
        {
            this->applyConstraintType(aMatrix, aVector, mStateDotDotBcDofs, mStateDotDotBcValues);
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    applyConstraintType(
      const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
      const Plato::ScalarVector                & aVector,
      const Plato::OrdinalVector               & aBcDofs,
      const Plato::ScalarVector                & aBcValues
    )
    {
        if(mJacobianU->isBlockMatrix())
        {
            Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, aBcDofs, aBcValues);
        }
        else
        {
            Plato::applyConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, aBcDofs, aBcValues);
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution)
    { return; }

    template<typename PhysicsType>
    Plato::Solutions
    Problem<PhysicsType>::
    solution(const Plato::ScalarVector & aControl)
    {
        this->computeInitialState(aControl);

        Plato::Scalar tCurrentTime(0.0);
        for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++) {

            if (mUForm)
            {
                this->forwardStepUForm(aControl, tCurrentTime, tStepIndex );
            }
            else
            {
                this->forwardStepAForm(aControl, tCurrentTime, tStepIndex );
            }
        }

        auto tSolution = this->getSolution();
        return tSolution;
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    forwardStepUForm(
        const Plato::ScalarVector & aControl,
              Plato::Scalar       & aCurrentTime,
              Plato::OrdinalType    aStepIndex
    )
    {
        aCurrentTime += mTimeStep;

        Plato::ScalarVector tDisplacementPrev = Kokkos::subview(mDisplacement, aStepIndex-1, Kokkos::ALL());
        Plato::ScalarVector tVelocityPrev     = Kokkos::subview(mVelocity,     aStepIndex-1, Kokkos::ALL());
        Plato::ScalarVector tAccelerationPrev = Kokkos::subview(mAcceleration, aStepIndex-1, Kokkos::ALL());

        Plato::ScalarVector tDisplacement = Kokkos::subview(mDisplacement, aStepIndex, Kokkos::ALL());
        Plato::ScalarVector tVelocity     = Kokkos::subview(mVelocity,     aStepIndex, Kokkos::ALL());
        Plato::ScalarVector tAcceleration = Kokkos::subview(mAcceleration, aStepIndex, Kokkos::ALL());

        // -R
        auto tResidual = mPDEConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);
        Plato::blas1::scale(-1.0, tResidual);

        // R_{v}
        auto tResidualV = mIntegrator->v_value(tDisplacement, tDisplacementPrev,
                                               tVelocity,     tVelocityPrev,
                                               tAcceleration, tAccelerationPrev, mTimeStep);

        // R_{,v^N}
        mJacobianV = mPDEConstraint.gradient_v(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);

        // -R += R_{,v^N} R_{v}
        Plato::MatrixTimesVectorPlusVector(mJacobianV, tResidualV, tResidual);

        // R_{a}
        auto tResidualA = mIntegrator->a_value(tDisplacement, tDisplacementPrev,
                                               tVelocity,     tVelocityPrev,
                                               tAcceleration, tAccelerationPrev, mTimeStep);

        // R_{,a^N}
        mJacobianA = mPDEConstraint.gradient_a(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);

        // -R += R_{,a^N} R_{a}
        Plato::MatrixTimesVectorPlusVector(mJacobianA, tResidualA, tResidual);

        // R_{,u^N}
        mJacobianU = mPDEConstraint.gradient_u(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);

        // R_{v,u^N}
        auto tR_vu = mIntegrator->v_grad_u(mTimeStep);

        // R_{,u^N} += R_{,v^N} R_{v,u^N}
        Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

        // R_{a,u^N}
        auto tR_au = mIntegrator->a_grad_u(mTimeStep);

        // R_{,u^N} += R_{,a^N} R_{a,u^N}
        Plato::blas1::axpy(-tR_au, mJacobianA->entries(), mJacobianU->entries());

        mStateBoundaryConditions.get(mStateBcDofs, mStateBcValues, aCurrentTime);
        this->applyConstraints(mJacobianU, tResidual);

        Plato::ScalarVector tDeltaD("increment", tDisplacement.extent(0));
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);

        // compute displacement increment:
        mSolver->solve(*mJacobianU, tDeltaD, tResidual);

        // compute and add velocity increment: \Delta v = - ( R_{v} + R_{v,u} \Delta u )
        Plato::blas1::axpy(tR_vu, tDeltaD, tResidualV);
        // v_{k+1} = v_{k} + \Delta v
        Plato::blas1::axpy(-1.0, tResidualV, tVelocity);

        // compute and add acceleration increment: \Delta a = - ( R_{a} + R_{a,u} \Delta u )
        Plato::blas1::axpy(tR_au, tDeltaD, tResidualA);
        // a_{k+1} = a_{k} + \Delta a
        Plato::blas1::axpy(-1.0, tResidualA, tAcceleration);

        // add displacement increment
        Plato::blas1::axpy(1.0, tDeltaD, tDisplacement);

        // fill in essential boundary fields
        this->constrainFieldsAtBoundary(tDisplacement,tVelocity,tAcceleration,aCurrentTime);

        if ( mSaveState )
        {
            // evaluate at new state
            tResidual  = mPDEConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);
            mDataMap.saveState();
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    forwardStepAForm(
        const Plato::ScalarVector & aControl,
              Plato::Scalar       & aCurrentTime,
              Plato::OrdinalType    aStepIndex
    )
    {
        aCurrentTime += mTimeStep;

        Plato::ScalarVector tDisplacementPrev = Kokkos::subview(mDisplacement, aStepIndex-1, Kokkos::ALL());
        Plato::ScalarVector tVelocityPrev     = Kokkos::subview(mVelocity,     aStepIndex-1, Kokkos::ALL());
        Plato::ScalarVector tAccelerationPrev = Kokkos::subview(mAcceleration, aStepIndex-1, Kokkos::ALL());

        Plato::ScalarVector tDisplacement = Kokkos::subview(mDisplacement, aStepIndex, Kokkos::ALL());
        Plato::ScalarVector tVelocity     = Kokkos::subview(mVelocity,     aStepIndex, Kokkos::ALL());
        Plato::ScalarVector tAcceleration = Kokkos::subview(mAcceleration, aStepIndex, Kokkos::ALL());

        // -R
        auto tResidual  = mPDEConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);
        Plato::blas1::scale(-1.0, tResidual);

        // R_{v}
        auto tResidualV = mIntegrator->v_value(tDisplacement, tDisplacementPrev,
                                                tVelocity,     tVelocityPrev,
                                                tAcceleration, tAccelerationPrev, mTimeStep);

        // R_{,v^N}
        mJacobianV = mPDEConstraint.gradient_v(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);

        // -R += R_{,v^N} R_{v}
        Plato::MatrixTimesVectorPlusVector(mJacobianV, tResidualV, tResidual);

        // R_{u}
        auto tResidualU = mIntegrator->u_value(tDisplacement, tDisplacementPrev,
                                                tVelocity,     tVelocityPrev,
                                                tAcceleration, tAccelerationPrev, mTimeStep);

        // R_{,u^N}
        mJacobianU = mPDEConstraint.gradient_u(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);

        // -R += R_{,u^N} R_{u}
        Plato::MatrixTimesVectorPlusVector(mJacobianU, tResidualU, tResidual);

        // R_{,a^N}
        mJacobianA = mPDEConstraint.gradient_a(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);

        // R_{v,a^N}
        auto tR_va = mIntegrator->v_grad_a(mTimeStep);

        // R_{,a^N} -= R_{,v^N} R_{v,a^N}
        Plato::blas1::axpy(-tR_va, mJacobianV->entries(), mJacobianA->entries());

        // R_{u,a^N}
        auto tR_ua = mIntegrator->u_grad_a(mTimeStep);

        // R_{,a^N} -= R_{,u^N} R_{u,a^N}
        Plato::blas1::axpy(-tR_ua, mJacobianU->entries(), mJacobianA->entries());

        mStateDotDotBoundaryConditions.get(mStateDotDotBcDofs, mStateDotDotBcValues, aCurrentTime);
        this->applyConstraints(mJacobianA, tResidual);

        Plato::ScalarVector tDeltaA("increment", tAcceleration.extent(0));
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaA);

        // compute acceleration increment:
        if(tR_ua == 0.0)
        {
          Plato::Solve::RowSummed<ElementType::mNumDofsPerNode>(mJacobianA, tDeltaA, tResidual);
        } else {
          mSolver->solve(*mJacobianA, tDeltaA, tResidual);
        }

        // compute and add velocity increment: \Delta v = - ( R_{v} + R_{v,a} \Delta a )
        Plato::blas1::axpy(tR_va, tDeltaA, tResidualV);
        // v_{k+1} = v_{k} + \Delta v
        Plato::blas1::axpy(-1.0, tResidualV, tVelocity);

        // compute and add displacement increment: \Delta u = - ( R_{u} + R_{u,a} \Delta a )
        Plato::blas1::axpy(tR_ua, tDeltaA, tResidualU);
        // u_{k+1} = u_{k} + \Delta u
        Plato::blas1::axpy(-1.0, tResidualU, tDisplacement);

        // add acceleration increment
        Plato::blas1::axpy(1.0, tDeltaA, tAcceleration);

        // fill in essential boundary fields
        this->constrainFieldsAtBoundary(tDisplacement,tVelocity,tAcceleration,aCurrentTime);

        if ( mSaveState )
        {
            // evaluate at new state
            tResidual  = mPDEConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep, aCurrentTime);
            mDataMap.saveState();
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    computeInitialState(
        const Plato::ScalarVector & aControl
    )
    {
        mDataMap.clearStates();
        mDataMap.scalarNodeFields["Topology"] = aControl;

        Kokkos::deep_copy(mDisplacement, 0.0);
        Kokkos::deep_copy(mVelocity,     0.0);
        Kokkos::deep_copy(mAcceleration, 0.0);
        Plato::ScalarVector tDisplacementInit = Kokkos::subview(mDisplacement, /*StepIndex=*/0, Kokkos::ALL());
        Plato::ScalarVector tVelocityInit     = Kokkos::subview(mVelocity,     /*StepIndex=*/0, Kokkos::ALL());
        Plato::ScalarVector tAccelerationInit = Kokkos::subview(mAcceleration, /*StepIndex=*/0, Kokkos::ALL());
        Kokkos::deep_copy(tDisplacementInit, mInitDisplacement);
        Kokkos::deep_copy(tVelocityInit,     mInitVelocity);
        Kokkos::deep_copy(tAccelerationInit, mInitAcceleration);

        this->constrainFieldsAtBoundary(tDisplacementInit,tVelocityInit,tAccelerationInit,0.0);

        auto tResidual = mPDEConstraint.value(tDisplacementInit, tVelocityInit, tAccelerationInit, aControl, mTimeStep, 0.0);
        mDataMap.saveState();
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    constrainFieldsAtBoundary(
              Plato::ScalarVector & aDisplacement,
              Plato::ScalarVector & aVelocity,
              Plato::ScalarVector & aAcceleration,
        const Plato::Scalar         aTime)
    {
        if (mUForm)
        {
            this->constrainUFormFieldsAtBoundary(aVelocity, aAcceleration, aTime);
        }
        else
        {
            this->constrainAFormFieldsAtBoundary(aDisplacement, aVelocity, aTime);
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    constrainUFormFieldsAtBoundary(
              Plato::ScalarVector & aVelocity,
              Plato::ScalarVector & aAcceleration,
        const Plato::Scalar         aTime)
    {
        mStateDotBoundaryConditions.get(mStateDotBcDofs, mStateDotBcValues, aTime);
        mStateDotDotBoundaryConditions.get(mStateDotDotBcDofs, mStateDotDotBcValues, aTime);
        if (mStateDotBcDofs.size() > 0)
        {
            Plato::enforce_boundary_condition(mStateDotBcDofs, mStateDotBcValues, aVelocity);
        }
        if (mStateDotDotBcDofs.size() > 0)
        {
            Plato::enforce_boundary_condition(mStateDotDotBcDofs, mStateDotDotBcValues, aAcceleration);
        }
    }

    template<typename PhysicsType>
    void
    Problem<PhysicsType>::
    constrainAFormFieldsAtBoundary(
              Plato::ScalarVector & aDisplacement,
              Plato::ScalarVector & aVelocity,
        const Plato::Scalar         aTime)
    {
        mStateBoundaryConditions.get(mStateBcDofs, mStateBcValues, aTime);
        mStateDotBoundaryConditions.get(mStateDotBcDofs, mStateDotBcValues, aTime);
        if (mStateBcDofs.size() > 0)
        {
            Plato::enforce_boundary_condition(mStateBcDofs, mStateBcValues, aDisplacement);
        }
        if (mStateDotBcDofs.size() > 0)
        {
            Plato::enforce_boundary_condition(mStateDotBcDofs, mStateDotBcValues, aVelocity);
        }
    }
    
    template<typename PhysicsType>
    Plato::Scalar
    Problem<PhysicsType>::
    criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) 
    {
        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            return tCriterion->value(aSolution, aControl, mTimeStep);
        }
        else
        {
            ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    template<typename PhysicsType>
    Plato::Scalar
    Problem<PhysicsType>::
    criterionValue(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )
    {
        if( mCriteria.count(aName) )
        {
            auto tSolution = this->getSolution();
            Criterion tCriterion = mCriteria[aName];
            return tCriterion->value(tSolution, aControl, mTimeStep);
        }
        else
        {
            ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )
    {
        if( mCriteria.count(aName) )
        {
            auto tSolution = this->getSolution();
            Criterion tCriterion = mCriteria[aName];
            return criterionGradient(aControl, tSolution, tCriterion);
        }
        else
        {
            ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    )
    {
        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            return criterionGradient(aControl, aSolution, tCriterion);
        }
        else
        {
            ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::
    criterionGradient(
      const Plato::ScalarVector & aControl,
      const Plato::Solutions    & aSolution,
            Criterion             aCriterion
    )
    {
        if(aCriterion == nullptr)
        {
            ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }

        Plato::Solutions tSolution(mPhysics);
        tSolution.set("State", mDisplacement);
        tSolution.set("StateDot", mVelocity);
        tSolution.set("StateDotDot", mAcceleration);

        // F_{,z}
        auto t_dFdz = aCriterion->gradient_z(tSolution, aControl, mTimeStep);

        auto tLastStepIndex = mNumSteps - 1;
        Plato::Scalar tCurrentTime(mTimeStep*mNumSteps);
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--)
        {
            auto tU = Kokkos::subview(mDisplacement, tStepIndex, Kokkos::ALL());
            auto tV = Kokkos::subview(mVelocity,     tStepIndex, Kokkos::ALL());
            auto tA = Kokkos::subview(mAcceleration, tStepIndex, Kokkos::ALL());

            Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
            Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());
            Plato::ScalarVector tAdjoint_A = Kokkos::subview(mAdjoints_A, tStepIndex, Kokkos::ALL());

            // F_{,u^k}
            auto t_dFdu = aCriterion->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
            // F_{,v^k}
            auto t_dFdv = aCriterion->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);
            // F_{,a^k}
            auto t_dFda = aCriterion->gradient_a(tSolution, aControl, tStepIndex, mTimeStep);

            if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                // L_{v}^{k+1}
                Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                // L_{a}^{k+1}
                Plato::ScalarVector tAdjoint_A_next = Kokkos::subview(mAdjoints_A, tStepIndex+1, Kokkos::ALL());


                // R_{v,u^k}^{k+1}
                auto tR_vu_prev = mIntegrator->v_grad_u_prev(mTimeStep);

                // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);

                // R_{a,u^k}^{k+1}
                auto tR_au_prev = mIntegrator->a_grad_u_prev(mTimeStep);

                // F_{,u^k} += L_{a}^{k+1} R_{a,u^k}^{k+1}
                Plato::blas1::axpy(tR_au_prev, tAdjoint_A_next, t_dFdu);


                // R_{v,v^k}^{k+1}
                auto tR_vv_prev = mIntegrator->v_grad_v_prev(mTimeStep);

                // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                // R_{a,v^k}^{k+1}
                auto tR_av_prev = mIntegrator->a_grad_v_prev(mTimeStep);

                // F_{,v^k} += L_{a}^{k+1} R_{a,v^k}^{k+1}
                Plato::blas1::axpy(tR_av_prev, tAdjoint_A_next, t_dFdv);


                // R_{v,a^k}^{k+1}
                auto tR_va_prev = mIntegrator->v_grad_a_prev(mTimeStep);

                // F_{,a^k} += L_{v}^{k+1} R_{v,a^k}^{k+1}
                Plato::blas1::axpy(tR_va_prev, tAdjoint_V_next, t_dFda);

                // R_{a,a^k}^{k+1}
                auto tR_aa_prev = mIntegrator->a_grad_a_prev(mTimeStep);

                // F_{,a^k} += L_{a}^{k+1} R_{a,a^k}^{k+1}
                Plato::blas1::axpy(tR_aa_prev, tAdjoint_A_next, t_dFda);

            }
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

            // R_{v,u^k}
            auto tR_vu = mIntegrator->v_grad_u(mTimeStep);

            // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
            Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);

            // R_{a,u^k}
            auto tR_au = mIntegrator->a_grad_u(mTimeStep);

            // -F_{,u^k} += R_{a,u^k}^k F_{,a^k}
            Plato::blas1::axpy(tR_au, t_dFda, t_dFdu);

            // R_{u,u^k}
            mJacobianU = mPDEConstraint.gradient_u(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

            // R_{u,v^k}
            mJacobianV = mPDEConstraint.gradient_v(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

            // R_{u,a^k}
            mJacobianA = mPDEConstraint.gradient_a(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

            // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
            Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

            // R_{u,u^k} -= R_{a,u^k} R_{u,a^k}
            Plato::blas1::axpy(-tR_au, mJacobianA->entries(), mJacobianU->entries());

            this->applyConstraints(mJacobianU, t_dFdu);

            // L_u^k
            mSolver->solve(*mJacobianU, tAdjoint_U, t_dFdu);

            // L_v^k
            Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
            Plato::blas1::fill(0.0, tAdjoint_V);
            Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

            // L_a^k
            Plato::MatrixTimesVectorPlusVector(mJacobianA, tAdjoint_U, t_dFda);
            Plato::blas1::fill(0.0, tAdjoint_A);
            Plato::blas1::axpy(-1.0, t_dFda, tAdjoint_A);

            // R^k_{,z}
            auto t_dRdz = mPDEConstraint.gradient_z(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

            // F_{,z} += L_u^k R^k_{,z}
            Plato::MatrixTimesVectorPlusVector(t_dRdz, tAdjoint_U, t_dFdz);

            tCurrentTime -= mTimeStep;
        }

        return t_dFdz;
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )
    {
        if( mCriteria.count(aName) )
        {
            auto tCriterion = mCriteria[aName];
            auto tSolution = this->getSolution();
            return criterionGradientX(aControl, tSolution, tCriterion);
        }
        else
        {
            ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    )
    {
        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            return criterionGradientX(aControl, aSolution, tCriterion);
        }
        else
        {
            ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    template<typename PhysicsType>
    Plato::ScalarVector
    Problem<PhysicsType>::
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion
    )
    {
        if(aCriterion == nullptr)
        {
            ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }

        Plato::Solutions tSolution(mPhysics);
        tSolution.set("State", mDisplacement);
        tSolution.set("StateDot", mVelocity);
        tSolution.set("StateDotDot", mAcceleration);

        // F_{,x}
        auto t_dFdx = aCriterion->gradient_x(tSolution, aControl, mTimeStep);

        auto tLastStepIndex = mNumSteps - 1;
        Plato::Scalar tCurrentTime(mTimeStep*mNumSteps);
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

            auto tU = Kokkos::subview(mDisplacement, tStepIndex, Kokkos::ALL());
            auto tV = Kokkos::subview(mVelocity,     tStepIndex, Kokkos::ALL());
            auto tA = Kokkos::subview(mAcceleration, tStepIndex, Kokkos::ALL());

            Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
            Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());
            Plato::ScalarVector tAdjoint_A = Kokkos::subview(mAdjoints_A, tStepIndex, Kokkos::ALL());

            // F_{,u^k}
            auto t_dFdu = aCriterion->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
            // F_{,v^k}
            auto t_dFdv = aCriterion->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);
            // F_{,a^k}
            auto t_dFda = aCriterion->gradient_a(tSolution, aControl, tStepIndex, mTimeStep);

            if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                // L_{v}^{k+1}
                Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                // L_{a}^{k+1}
                Plato::ScalarVector tAdjoint_A_next = Kokkos::subview(mAdjoints_A, tStepIndex+1, Kokkos::ALL());


                // R_{v,u^k}^{k+1}
                auto tR_vu_prev = mIntegrator->v_grad_u_prev(mTimeStep);

                // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);

                // R_{a,u^k}^{k+1}
                auto tR_au_prev = mIntegrator->a_grad_u_prev(mTimeStep);

                // F_{,u^k} += L_{a}^{k+1} R_{a,u^k}^{k+1}
                Plato::blas1::axpy(tR_au_prev, tAdjoint_A_next, t_dFdu);


                // R_{v,v^k}^{k+1}
                auto tR_vv_prev = mIntegrator->v_grad_v_prev(mTimeStep);

                // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                // R_{a,v^k}^{k+1}
                auto tR_av_prev = mIntegrator->a_grad_v_prev(mTimeStep);

                // F_{,v^k} += L_{a}^{k+1} R_{a,v^k}^{k+1}
                Plato::blas1::axpy(tR_av_prev, tAdjoint_A_next, t_dFdv);


                // R_{v,a^k}^{k+1}
                auto tR_va_prev = mIntegrator->v_grad_a_prev(mTimeStep);

                // F_{,a^k} += L_{v}^{k+1} R_{v,a^k}^{k+1}
                Plato::blas1::axpy(tR_va_prev, tAdjoint_V_next, t_dFda);

                // R_{a,a^k}^{k+1}
                auto tR_aa_prev = mIntegrator->a_grad_a_prev(mTimeStep);

                // F_{,a^k} += L_{a}^{k+1} R_{a,a^k}^{k+1}
                Plato::blas1::axpy(tR_aa_prev, tAdjoint_A_next, t_dFda);

            }
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

            // R_{v,u^k}
            auto tR_vu = mIntegrator->v_grad_u(mTimeStep);

            // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
            Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);

            // R_{a,u^k}
            auto tR_au = mIntegrator->a_grad_u(mTimeStep);

            // -F_{,u^k} += R_{a,u^k}^k F_{,a^k}
            Plato::blas1::axpy(tR_au, t_dFda, t_dFdu);

            // R_{u,u^k}
            mJacobianU = mPDEConstraint.gradient_u(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

            // R_{u,v^k}
            mJacobianV = mPDEConstraint.gradient_v(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

            // R_{u,a^k}
            mJacobianA = mPDEConstraint.gradient_a(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

            // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
            Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

            // R_{u,u^k} -= R_{a,u^k} R_{u,a^k}
            Plato::blas1::axpy(-tR_au, mJacobianA->entries(), mJacobianU->entries());

            this->applyConstraints(mJacobianU, t_dFdu);

            // L_u^k
            mSolver->solve(*mJacobianU, tAdjoint_U, t_dFdu);

            // L_v^k
            Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
            Plato::blas1::fill(0.0, tAdjoint_V);
            Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

            // L_a^k
            Plato::MatrixTimesVectorPlusVector(mJacobianA, tAdjoint_U, t_dFda);
            Plato::blas1::fill(0.0, tAdjoint_A);
            Plato::blas1::axpy(-1.0, t_dFda, tAdjoint_A);

            // R^k_{,x}
            auto t_dRdx = mPDEConstraint.gradient_x(tU, tV, tA, aControl, mTimeStep, tCurrentTime);

            // F_{,x} += L_u^k R^k_{,x}
            Plato::MatrixTimesVectorPlusVector(t_dRdx, tAdjoint_U, t_dFdx);

            tCurrentTime -= mTimeStep;
        }

        return t_dFdx;
    }

    template<typename PhysicsType>
    Plato::Solutions
    Problem<PhysicsType>::
    getSolution() const
    {
        Plato::Solutions tSolution(mPhysics, mPDE);
        tSolution.set("State",       mDisplacement, mPDEConstraint.getDofNames());
        tSolution.set("StateDot",    mVelocity,     mPDEConstraint.getDofDotNames());
        tSolution.set("StateDotDot", mAcceleration, mPDEConstraint.getDofDotDotNames());
        return tSolution;
    }
}

}
