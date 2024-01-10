#pragma once

#include "BLAS1.hpp"

#include "PlatoMesh.hpp"
#include "ParseTools.hpp"
#include "Geometrical.hpp"
#include "EssentialBCs.hpp"
#include "AnalyzeOutput.hpp"
#include "AnalyzeMacros.hpp"
#include "ApplyConstraints.hpp"
#include "alg/PlatoAbstractSolver.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "parabolic/ScalarFunctionBaseFactory.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"

namespace Plato
{

namespace Parabolic
{

        /******************************************************************************/
        template<typename PhysicsType>
        Problem<PhysicsType>::
        Problem(
          Plato::Mesh              aMesh,
          Teuchos::ParameterList & aProblemParams,
          Comm::Machine            aMachine
        ) :
            AbstractProblem(aMesh, aProblemParams),
            mSpatialModel  (aMesh, aProblemParams, mDataMap),
            mPDEConstraint (mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint")),
            mTrapezoidIntegrator (aProblemParams.sublist("Time Integration")),
            mNumSteps      (Plato::ParseTools::getSubParam<int>   (aProblemParams, "Time Integration", "Number Time Steps",   1  )),
            mTimeStep      (Plato::ParseTools::getSubParam<Plato::Scalar>(aProblemParams, "Time Integration", "Time Step" ,   1.0)),
            mNumNewtonSteps(Plato::ParseTools::getSubParam<int>   (aProblemParams, "Newton Iteration", "Maximum Iterations",  1  )),
            mNewtonIncTol  (Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Increment Tolerance", 0.0)),
            mNewtonResTol  (Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Residual Tolerance",  0.0)),
            mSaveState     (aProblemParams.sublist("Parabolic").isType<Teuchos::Array<std::string>>("Plottable")),
            mResidual      ("MyResidual", mPDEConstraint.size()),
            mState         ("State",      mNumSteps, mPDEConstraint.size()),
            mStateDot      ("StateDot",   mNumSteps, mPDEConstraint.size()),
            mJacobianU     (Teuchos::null),
            mJacobianV     (Teuchos::null),
            mPDE           (aProblemParams.get<std::string>("PDE Constraint")),
            mPhysics       (aProblemParams.get<std::string>("Physics")),
            mMPCs          (nullptr)
        /******************************************************************************/
        {
            // parse criteria
            //
            if(aProblemParams.isSublist("Criteria"))
            {
                Plato::Geometric::ScalarFunctionBaseFactory<Plato::Geometrical<TopoElementType>> tLinearFunctionBaseFactory;
                Plato::Parabolic::ScalarFunctionBaseFactory<PhysicsType> tFunctionBaseFactory;

                auto tCriteriaParams = aProblemParams.sublist("Criteria");
                for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
                {
                    const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
                    std::string tName = tCriteriaParams.name(tIndex);

                    TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
                      " Parameter in Criteria block not valid.  Expect lists only.");

                    if( tCriteriaParams.sublist(tName).get<bool>("Linear", false) == true )
                    {
                        auto tCriterion = tLinearFunctionBaseFactory.create(mSpatialModel, mDataMap, aProblemParams, tName);
                        if( tCriterion != nullptr )
                        {
                            mLinearCriteria[tName] = tCriterion;
                        }
                    }
                    else
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
                }
            }

            if(aProblemParams.isSublist("Multipoint Constraints") == true)
            {
                Plato::OrdinalType tNumDofsPerNode = mPDEConstraint.numDofsPerNode();
                auto & tMyParams = aProblemParams.sublist("Multipoint Constraints", false);
                mMPCs = std::make_shared<Plato::MultipointConstraints>(mSpatialModel, tNumDofsPerNode, tMyParams);
                mMPCs->setupTransform();
            }

            // parse boundary constraints
            //
            Plato::EssentialBCs<ElementType>
                tEssentialBoundaryConditions(aProblemParams.sublist("Essential Boundary Conditions",false), mSpatialModel.Mesh);
            tEssentialBoundaryConditions.get(mStateBcDofs, mStateBcValues);

            if(mMPCs)
            {
                mMPCs->checkEssentialBcsConflicts(mStateBcDofs);
            }

            // parse computed fields
            //
            if(aProblemParams.isSublist("Computed Fields"))
            {
              mComputedFields = Teuchos::rcp(new Plato::ComputedFields<ElementType::mNumSpatialDims>(aMesh, aProblemParams.sublist("Computed Fields")));
            }

            // parse initial state
            //
            if(aProblemParams.isSublist("Initial State"))
            {
                Plato::ScalarVector tInitialState = Kokkos::subview(mState, 0, Kokkos::ALL());
                if(mComputedFields == Teuchos::null) {
                  ANALYZE_THROWERR("No 'Computed Fields' have been defined");
                }

                auto tDofNames = mPDEConstraint.getDofNames();
    
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
                            if (Plato::tolower(tDofNames[j]) == Plato::tolower(tName)) {
                               tDofIndex = j;
                            }
                        }
                        if (tDofIndex == -1)
                        {
                          std::stringstream ss;
                          ss << "Tried to initialize non-existent state field: " << Plato::tolower(tName) << std::endl;
                          ss << "Available states are: " << std::endl;
                          for (const auto & tDofName : tDofNames)
                          {
                            ss << "  " << Plato::tolower(tDofName) << std::endl;
                          }
                          ANALYZE_THROWERR(ss.str());
                        }
                        mComputedFields->get(tFieldName, tDofIndex, tDofNames.size(), tInitialState);
                    }
                }
            }

            Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"), LinearSystemType::SYMMETRIC_INDEFINITE);
            mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode, mMPCs);

        }

        /******************************************************************************//**
         * \brief Is criterion independent of the solution state?
         * \param [in] aName Name of criterion.
        **********************************************************************************/
        template<typename PhysicsType>
        bool
        Problem<PhysicsType>::
        criterionIsLinear(
            const std::string & aName
        )
        {
            return mLinearCriteria.count(aName) > 0 ? true : false;
        }

        template<typename PhysicsType>
        void
        Problem<PhysicsType>::
        applyConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector
        ){}

        /******************************************************************************/
        template<typename PhysicsType>
        void
        Problem<PhysicsType>::
        applyStateConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector,
          Plato::Scalar aScale
        )
        /******************************************************************************/
        {
            if(mJacobianU->isBlockMatrix())
            {
                Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, mStateBcDofs, mStateBcValues, aScale);
            }
            else
            {
                Plato::applyConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, mStateBcDofs, mStateBcValues, aScale);
            }
        }

        /******************************************************************************/ /**
        * \brief Output solution to visualization file.
        * \param [in] aFilepath output/visualizaton file path
        **********************************************************************************/
        template<typename PhysicsType>
        void
        Problem<PhysicsType>::
        output(const std::string &aFilepath)
        {
            auto tDataMap = this->getDataMap();
            auto tSolution = this->getSolution();
            auto tSolutionOutput = mPDEConstraint.getSolutionStateOutputData(tSolution);
            Plato::universal_solution_output(aFilepath, tSolutionOutput, tDataMap, mSpatialModel.Mesh);
        }

        /******************************************************************************//**
         * \brief Update physics-based parameters within optimization iterations
         * \param [in] aState 2D container of state variables
         * \param [in] aControl 1D container of control variables
        **********************************************************************************/
        template<typename PhysicsType>
        void
        Problem<PhysicsType>::
        updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution)
        { return; }

        /******************************************************************************/
        template<typename PhysicsType>
        Plato::Solutions
        Problem<PhysicsType>::
        solution(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {

            mDataMap.clearStates();

            mDataMap.scalarNodeFields["Topology"] = aControl;
            Plato::ScalarVector tStateInit    = Kokkos::subview(mState,    /*StepIndex=*/0, Kokkos::ALL());
            Plato::ScalarVector tStateDotInit = Kokkos::subview(mStateDot, /*StepIndex=*/0, Kokkos::ALL());
            mResidual  = mPDEConstraint.value(tStateInit, tStateDotInit, aControl, mTimeStep);
            mDataMap.saveState();
   
            for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++) {
              Plato::ScalarVector tStatePrev    = Kokkos::subview(mState,    tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tStateDotPrev = Kokkos::subview(mStateDot, tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tState        = Kokkos::subview(mState,    tStepIndex,   Kokkos::ALL());
              Plato::ScalarVector tStateDot     = Kokkos::subview(mStateDot, tStepIndex,   Kokkos::ALL());

              // inner loop for non-linear models
              for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
              {
                  // -R_{u}
                  mResidual  = mPDEConstraint.value(tState, tStateDot, aControl, mTimeStep);
                  Plato::blas1::scale(-1.0, mResidual);

                  // R_{v}
                  mResidualV = mTrapezoidIntegrator.v_value(tState,    tStatePrev,
                                                            tStateDot, tStateDotPrev, mTimeStep);

                  // R_{u,v^N}
                  mJacobianV = mPDEConstraint.gradient_v(tState, tStateDot, aControl, mTimeStep);

                  // -R_{u} += R_{u,v^N} R_{v}
                  Plato::MatrixTimesVectorPlusVector(mJacobianV, mResidualV, mResidual);

                  // R_{u,u^N}
                  mJacobianU = mPDEConstraint.gradient_u(tState, tStateDot, aControl, mTimeStep);

                  // R_{v,u^N}
                  auto tR_vu = mTrapezoidIntegrator.v_grad_u(mTimeStep);

                  // R_{u,u^N} += R_{u,v^N} R_{v,u^N}
                  Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

                  Plato::OrdinalType tScale = (tNewtonIndex == 0) ? 1.0 : 0.0;
                  this->applyStateConstraints(mJacobianU, mResidual, tScale);

                  if (mNumNewtonSteps > 1) {
                      auto tResidualNorm = Plato::blas1::norm(mResidual);
                      std::cout << " Residual norm: " << tResidualNorm << std::endl;
                      if (tResidualNorm < mNewtonResTol) {
                          std::cout << " Residual norm tolerance satisfied." << std::endl;
                          break;
                      }
                  }

                  Plato::ScalarVector tDeltaD("increment", tState.extent(0));
                  Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);

                  // compute displacement increment:
                  mSolver->solve(*mJacobianU, tDeltaD, mResidual);

                  // compute and add statedot increment: \Delta v = - ( R_{v} + R_{v,u} \Delta u )
                  Plato::blas1::axpy(tR_vu, tDeltaD, mResidualV);

                  // a_{k+1} = a_{k} + \Delta a
                  Plato::blas1::axpy(-1.0, mResidualV, tStateDot);

                  // add displacement increment
                  Plato::blas1::axpy(1.0, tDeltaD, tState);

                  if (mNumNewtonSteps > 1) {
                      auto tIncrementNorm = Plato::blas1::norm(tDeltaD);
                      std::cout << " Delta norm: " << tIncrementNorm << std::endl;
                      if (tIncrementNorm < mNewtonIncTol) {
                          std::cout << " Solution increment norm tolerance satisfied." << std::endl;
                          break;
                      }
                  }
              }

              if ( mSaveState )
              {
                // evaluate at new state
                mResidual  = mPDEConstraint.value(tState, tStateDot, aControl, mTimeStep);
                mDataMap.saveState();
              }
            }

            auto tSolution = this->getSolution();
            return tSolution;
        }

        /******************************************************************************//**
         * \brief Evaluate criterion function
         * \param [in] aControl 1D view of control variables
         * \param [in] aName Name of criterion.
         * \return criterion function value
        **********************************************************************************/
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
                return tCriterion->value(tSolution, aControl);
            }
            else
            if( mLinearCriteria.count(aName) )
            {
                LinearCriterion tCriterion = mLinearCriteria[aName];
                return tCriterion->value(aControl);
            }
            else
            {
                ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }


        /******************************************************************************//**
         * \brief Evaluate criterion function
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aName Name of criterion.
         * \return criterion function value
        **********************************************************************************/
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
            if( mLinearCriteria.count(aName) )
            {
                LinearCriterion tCriterion = mLinearCriteria[aName];
                return tCriterion->value(aControl);
            }
            else
            {
                ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion partial derivative wrt control variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aName Name of criterion.
         * \return 1D view - criterion partial derivative wrt control variables
        **********************************************************************************/
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
                Criterion tCriterion = mCriteria[aName];
                auto tSolution = this->getSolution();
                return criterionGradient(aControl, tSolution, tCriterion);
            }
            else
            if( mLinearCriteria.count(aName) )
            {
                LinearCriterion tCriterion = mLinearCriteria[aName];
                return tCriterion->gradient_z(aControl);
            }
            else
            {
                ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt control variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aName Name of criterion.
         * \return 1D view - criterion gradient wrt control variables
        **********************************************************************************/
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
            if( mLinearCriteria.count(aName) )
            {
                LinearCriterion tCriterion = mLinearCriteria[aName];
                return tCriterion->gradient_z(aControl);
            }
            else
            {
                ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt control variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aCriterion criterion to be evaluated
         * \return 1D view - criterion gradient wrt control variables
        **********************************************************************************/
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
                ANALYZE_THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
            }

            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mState);
            tSolution.set("StateDot", mStateDot);

            // F_{,z}
            auto t_dFdz = aCriterion->gradient_z(tSolution, aControl, mTimeStep);

            auto tLastStepIndex = mNumSteps - 1;
            for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

                auto tU = Kokkos::subview(mState, tStepIndex, Kokkos::ALL());
                auto tV = Kokkos::subview(mStateDot, tStepIndex, Kokkos::ALL());

                Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());

                // F_{,u^k}
                auto t_dFdu = aCriterion->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,v^k}
                auto t_dFdv = aCriterion->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mTrapezoidIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mTrapezoidIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                }
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mTrapezoidIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);

                // R_{u,u^k}
                mJacobianU = mPDEConstraint.gradient_u_T(tU, tV, aControl, mTimeStep);

                // R_{u,v^k}
                mJacobianV = mPDEConstraint.gradient_v_T(tU, tV, aControl, mTimeStep);

                // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
                Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

                this->applyStateConstraints(mJacobianU, t_dFdu, /*scale_constraints_by*/0.0);

                // L_u^k
                mSolver->solve(*mJacobianU, tAdjoint_U, t_dFdu);

                // L_v^k
                Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
                Plato::blas1::fill(0.0, tAdjoint_V);
                Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

                // R^k_{,z}
                auto t_dRdz = mPDEConstraint.gradient_z(tU, tV, aControl, mTimeStep);

                // F_{,z} += L_u^k R^k_{,z}
                Plato::MatrixTimesVectorPlusVector(t_dRdz, tAdjoint_U, t_dFdz);
            }

            return t_dFdz;
        }


        /******************************************************************************//**
         * \brief Evaluate criterion partial derivative wrt configuration variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aName Name of criterion.
         * \return 1D view - criterion partial derivative wrt configuration variables
        **********************************************************************************/
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
                Criterion tCriterion = mCriteria[aName];
                auto tSolution = this->getSolution();
                return criterionGradientX(aControl, tSolution, tCriterion);
            }
            else
            if( mLinearCriteria.count(aName) )
            {
                LinearCriterion tCriterion = mLinearCriteria[aName];
                return tCriterion->gradient_x(aControl);
            }
            else
            {
                ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt configuration variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aSolution solution database
         * \param [in] aName Name of criterion.
         * \return 1D view - criterion gradient wrt control variables
        **********************************************************************************/
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
            if( mLinearCriteria.count(aName) )
            {
                LinearCriterion tCriterion = mLinearCriteria[aName];
                return tCriterion->gradient_x(aControl);
            }
            else
            {
                ANALYZE_THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
            }
        }

        /******************************************************************************//**
         * \brief Evaluate criterion gradient wrt configuration variables
         * \param [in] aControl 1D view of control variables
         * \param [in] aGlobalState 2D view of state variables
         * \param [in] aCriterion criterion to be evaluated
         * \return 1D view - criterion gradient wrt configuration variables
        **********************************************************************************/
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
                ANALYZE_THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
            }

            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mState);
            tSolution.set("StateDot", mStateDot);

            // F_{,x}
            auto t_dFdx = aCriterion->gradient_x(tSolution, aControl, mTimeStep);

            auto tLastStepIndex = mNumSteps - 1;
            for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

                auto tU = Kokkos::subview(mState, tStepIndex, Kokkos::ALL());
                auto tV = Kokkos::subview(mStateDot, tStepIndex, Kokkos::ALL());

                Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());

                // F_{,u^k}
                auto t_dFdu = aCriterion->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
                // F_{,v^k}
                auto t_dFdv = aCriterion->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());


                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mTrapezoidIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::blas1::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mTrapezoidIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::blas1::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                }
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mTrapezoidIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::blas1::axpy(tR_vu, t_dFdv, t_dFdu);


                // R_{u,u^k}
                mJacobianU = mPDEConstraint.gradient_u_T(tU, tV, aControl, mTimeStep);

                // R_{u,v^k}
                mJacobianV = mPDEConstraint.gradient_v_T(tU, tV, aControl, mTimeStep);

                // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
                Plato::blas1::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

                this->applyStateConstraints(mJacobianU, t_dFdu, /*scale_constraints_by*/0.0);

                // L_u^k
                mSolver->solve(*mJacobianU, tAdjoint_U, t_dFdu);

                // L_v^k
                Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
                Plato::blas1::fill(0.0, tAdjoint_V);
                Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

                // R^k_{,x}
                auto t_dRdx = mPDEConstraint.gradient_x(tU, tV, aControl, mTimeStep);

                // F_{,x} += L_u^k R^k_{,x}
                Plato::MatrixTimesVectorPlusVector(t_dRdx, tAdjoint_U, t_dFdx);
            }

            return t_dFdx;
        }

        /******************************************************************************//**
         * \brief Return solution database.
         * \return solution database
        **********************************************************************************/
        template<typename PhysicsType>
        Plato::Solutions
        Problem<PhysicsType>::
        getSolution() const
        {
            Plato::Solutions tSolution(mPhysics, mPDE);
            tSolution.set("State",    mState,    mPDEConstraint.getDofNames());
            tSolution.set("StateDot", mStateDot, mPDEConstraint.getDofDotNames());
            return tSolution;
        }
} // namespace Parabolic

} // namespace Plato
