#ifndef PLATO_PROBLEM_PARABOLIC_PROBLEM_DEF
#define PLATO_PROBLEM_PARABOLIC_PROBLEM_DEF

#include "boundary_conditions/ApplyConstraints.hpp"
#include "boundary_conditions/EssentialBCs.hpp"
#include "domain/AnalyzeOutput.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "mesh/ComputedField.hpp"
#include "mesh/PlatoMesh.hpp"
#include "parsing/ParseTools.hpp"
#include "parsing/TeuchosParsingUtilities.hpp"
#include "problem/Geometrical.hpp"
#include "problem/elliptic/ScalarFunctionBaseFactory.hpp"
#include "problem/geometric/ScalarFunctionBaseFactory.hpp"
#include "problem/parabolic/ScalarFunctionBaseFactory.hpp"
#include "solver/PlatoAbstractSolver.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "solver/nonlinear_solvers/NewtonSolver.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace plato::parabolic
{
template <typename PhysicsType>
Problem<PhysicsType>::Problem(Plato::Mesh aMesh, Teuchos::ParameterList& aProblemParams, Plato::Comm::Machine aMachine)
    : AbstractProblem(aMesh, aProblemParams),
      mSpatialModel(aMesh, plato::domain::parse_domains(aProblemParams, aMesh), mDataMap),
      mPDE(std::make_shared<VectorFunctionType>(mSpatialModel, mDataMap, aProblemParams)),
      mPDEType(aProblemParams.get<std::string>("PDE Constraint")),
      mPhysics(aProblemParams.get<std::string>("Physics")),
      mOutputFileStream(aProblemParams.isParameter("Output File") ? std::optional<std::ofstream>{std::ofstream{
                                                                        aProblemParams.get<std::string>("Output File")}}
                                                                  : std::nullopt),
      mOutputStream{mOutputFileStream.has_value() ? mOutputFileStream.value() : std::cout},
      mTrapezoidIntegrator(aProblemParams.sublist("Time Integration")),
      mNumSteps(Plato::ParseTools::getSubParam<int>(aProblemParams, "Time Integration", "Number Time Steps", 1)),
      mTimeStep(Plato::ParseTools::getSubParam<Plato::Scalar>(aProblemParams, "Time Integration", "Time Step", 1.0)),
      mNumNewtonSteps(Plato::ParseTools::getSubParam<int>(aProblemParams, "Newton Iteration", "Maximum Iterations", 1)),
      mNewtonResTol(
          Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Residual Tolerance", 0.0)),
      mNewtonIncTol(
          Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Increment Tolerance", 0.0)),
      mState("State", mNumSteps, mPDE->size()),
      mStateDot("StateDot", mNumSteps, mPDE->size()),
      mSaveState(aProblemParams.sublist("Parabolic").isType<Teuchos::Array<std::string>>("Plottable")),
      mEssentialBCs(aProblemParams.sublist("Essential Boundary Conditions", false), aMesh),
      mMPCs(nullptr)
{
    const auto tSystemType = mPhysics == "Thermomechanical" ? Plato::LinearSystemType::SYMMETRIC_PATTERN
                                                            : Plato::LinearSystemType::SYMMETRIC_INDEFINITE;
    auto tSolverFactory = Plato::SolverFactory{aProblemParams.sublist("Linear Solver"), tSystemType};
    mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode, mMPCs);

    if (aProblemParams.isSublist("Criteria"))
    {
        auto tAddCriteria = [&tCriteriaMap = mCriteriaMap, &tSpatialModel = mSpatialModel, &tDataMap = mDataMap,
                             &tProblemParams = aProblemParams,
                             tCriterionBaseFactory =
                                 Plato::Parabolic::ScalarFunctionBaseFactory<PhysicsType>{}](const std::string& aName)
        {
            const auto tCriterion = tCriterionBaseFactory.create(tSpatialModel, tDataMap, tProblemParams, aName);
            if (tCriterion)
            {
                tCriteriaMap[aName] = tCriterion;
            }
        };
        utilities::for_each_sublist(aProblemParams.sublist("Criteria"), tAddCriteria);

        if (mCriteriaMap.size())
        {
            const auto tLength = mPDE->size();
            mAdjointStates = Plato::ScalarMultiVector("Adjoint States", mNumSteps, tLength);
            mAdjointStatesV = Plato::ScalarMultiVector("Adjoint States V", mNumSteps, tLength);
        }
    }

    if (aProblemParams.isSublist("Multipoint Constraints") == true)
    {
        const Plato::OrdinalType tNumDofsPerNode = mPDE->numDofsPerNode();
        auto& tMyParams = aProblemParams.sublist("Multipoint Constraints", false);
        mMPCs = std::make_shared<Plato::MultipointConstraints>(mSpatialModel, tNumDofsPerNode, tMyParams);
        mMPCs->setupTransform();
    }
    if (mMPCs)
    {
        Plato::OrdinalVector tBcDofs;
        Plato::ScalarVector tBcValues;
        mEssentialBCs.get(tBcDofs, tBcValues);
        mMPCs->checkEssentialBcsConflicts(tBcDofs);
    }

    if (aProblemParams.isSublist("Initial State"))
    {
        if (!aProblemParams.isSublist("Computed Fields"))
        {
            ANALYZE_THROWERR("No 'Computed Fields' have been defined");
        }
        const auto tComputedFields = Teuchos::rcp(
            new Plato::ComputedFields<ElementType::mNumSpatialDims>(aMesh, aProblemParams.sublist("Computed Fields")));

        Plato::ScalarVector tInitialState = Kokkos::subview(mState, 0, Kokkos::ALL());

        const auto tDofNames = mPDE->getDofNames();

        auto tInitStateParams = aProblemParams.sublist("Initial State");
        for (auto i = tInitStateParams.begin(); i != tInitStateParams.end(); ++i)
        {
            const auto& tEntry = tInitStateParams.entry(i);
            const auto& tName = tInitStateParams.name(i);

            if (tEntry.isList())
            {
                auto& tStateList = tInitStateParams.sublist(tName);
                auto tFieldName = tStateList.get<std::string>("Computed Field");
                int tDofIndex = -1;
                for (int j = 0; j < tDofNames.size(); ++j)
                {
                    if (Plato::tolower(tDofNames[j]) == Plato::tolower(tName))
                    {
                        tDofIndex = j;
                    }
                }
                if (tDofIndex == -1)
                {
                    std::stringstream ss;
                    ss << "Tried to initialize non-existent state field: " << Plato::tolower(tName) << std::endl;
                    ss << "Available states are: " << std::endl;
                    for (const auto& tDofName : tDofNames)
                    {
                        ss << "  " << Plato::tolower(tDofName) << std::endl;
                    }
                    ANALYZE_THROWERR(ss.str());
                }
                tComputedFields->get(tFieldName, tDofIndex, tDofNames.size(), tInitialState);
            }
        }
    }
}

template <typename PhysicsType>
void Problem<PhysicsType>::applyStateConstraints(const Teuchos::RCP<Plato::CrsMatrixType>& aMatrix,
                                                 const Plato::ScalarVector& aVector,
                                                 Plato::Scalar aScale)
{
    Plato::OrdinalVector tStateBcDofs;
    Plato::ScalarVector tStateBcValues;
    mEssentialBCs.get(tStateBcDofs, tStateBcValues);
    if (aMatrix->isBlockMatrix())
    {
        Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, tStateBcDofs, tStateBcValues,
                                                                   aScale);
    }
    else
    {
        Plato::applyConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, tStateBcDofs, tStateBcValues, aScale);
    }
}

template <typename PhysicsType>
void Problem<PhysicsType>::output(const std::string& aFilepath)
{
    const auto tDataMap = this->getDataMap();
    const auto tSolution = this->getSolution();
    const auto tSolutionOutput = mPDE->getSolutionStateOutputData(tSolution);
    Plato::universal_solution_output(aFilepath, tSolutionOutput, tDataMap, mSpatialModel.mMesh);
}

template <typename PhysicsType>
void Problem<PhysicsType>::updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution)
{
}

template <typename PhysicsType>
Plato::Solutions Problem<PhysicsType>::solution(const Plato::ScalarVector& aControl)
{
    mDataMap.clearStates();

    const auto tNewtonSolver =
        algorithms::nonlinear_solvers::NewtonSolver{mNumNewtonSteps, mNewtonResTol, mNewtonIncTol, mSolver};

    Plato::ScalarVector tStateInit = Kokkos::subview(mState, /*StepIndex=*/0, Kokkos::ALL());
    Plato::ScalarVector tStateDotInit = Kokkos::subview(mStateDot, /*StepIndex=*/0, Kokkos::ALL());

    Plato::OrdinalVector tStateBcDofs;
    Plato::ScalarVector tStateBcValues;
    mEssentialBCs.get(tStateBcDofs, tStateBcValues);

    // TODO: compute initial state dot using implicit solve? Only if the initial temperature or thermal force is
    // non-zero

    mDataMap.scalarNodeFields["Topology"] = aControl;
    [[maybe_unused]] const auto tResidual = mPDE->value(tStateInit, tStateDotInit, aControl, mTimeStep);
    mDataMap.saveState();

    for (Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++)
    {
        mOutputStream << "\n Step " << std::to_string(tStepIndex) << "\n";

        Plato::ScalarVector tStatePrev = Kokkos::subview(mState, tStepIndex - 1, Kokkos::ALL());
        Plato::ScalarVector tStateDotPrev = Kokkos::subview(mStateDot, tStepIndex - 1, Kokkos::ALL());

        Plato::ScalarVector tState = Kokkos::subview(mState, tStepIndex, Kokkos::ALL());
        Plato::ScalarVector tStateDot = Kokkos::subview(mStateDot, tStepIndex, Kokkos::ALL());

        auto tComputeResidual = [&tPDE = mPDE, &tIntegrator = mTrapezoidIntegrator, &tStatePrev, &tStateDot,
                                 &tStateDotPrev, &tControl = aControl,
                                 tTimeStep = mTimeStep](const Plato::ScalarVector& aState) -> Plato::ScalarVector
        {
            // R_{u}
            const auto tResidual = tPDE->value(aState, tStateDot, tControl, tTimeStep);

            // R_{v}
            const auto tStateDotResidual = tIntegrator.v_value(aState, tStatePrev, tStateDot, tStateDotPrev, tTimeStep);
            Plato::blas1::scale(-1.0, tStateDotResidual);

            // R_{u,v^N}
            const auto tJacobianV = tPDE->gradient_v(aState, tStateDot, tControl, tTimeStep);

            // R_{u} -= R_{u,v^N} R_{v}
            Plato::MatrixTimesVectorPlusVector(tJacobianV, tStateDotResidual, tResidual);

            return tResidual;
        };

        auto tComputeJacobian = [&tPDE = mPDE, &tIntegrator = mTrapezoidIntegrator, &tStatePrev, &tStateDot,
                                 &tStateDotPrev, &tControl = aControl, tTimeStep = mTimeStep](
                                    const Plato::ScalarVector& aState) -> Teuchos::RCP<Plato::CrsMatrixType>
        {
            // R_{u,u^N}
            const auto tJacobian = tPDE->gradient_u(aState, tStateDot, tControl, tTimeStep);

            // R_{u,v^N}
            const auto tJacobianV = tPDE->gradient_v(aState, tStateDot, tControl, tTimeStep);

            // R_{u,u^N} -= R_{u,v^N} R_{v,u^N}
            Plato::blas1::axpy(-tIntegrator.v_grad_u(tTimeStep), tJacobianV->entries(), tJacobian->entries());

            return tJacobian;
        };

        auto tApplyBoundaryConditions =
            [tBcDofs = tStateBcDofs, tBcValues = tStateBcValues](
                Teuchos::RCP<Plato::CrsMatrixType>& aMatrix, Plato::ScalarVector& aVector, const Plato::Scalar aScale)
        {
            if (aMatrix->isBlockMatrix())
            {
                Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, tBcDofs, tBcValues,
                                                                           aScale);
            }
            else
            {
                Plato::applyConstraints<ElementType::mNumDofsPerNode>(aMatrix, aVector, tBcDofs, tBcValues, aScale);
            }
        };

        const bool tNewtonHasConverged =
            tNewtonSolver.solve(tState, tComputeResidual, tComputeJacobian, tApplyBoundaryConditions, mOutputStream);

        // update state dot
        Plato::blas1::axpy(-1.0, mTrapezoidIntegrator.v_value(tState, tStatePrev, tStateDot, tStateDotPrev, mTimeStep),
                           tStateDot);

        if (mSaveState)
        {
            // evaluate at new state
            [[maybe_unused]] const auto tResidual = mPDE->value(tState, tStateDot, aControl, mTimeStep);
            mDataMap.saveState();
        }
    }

    auto tSolution = this->getSolution();
    return tSolution;
}

template <typename PhysicsType>
Plato::Scalar Problem<PhysicsType>::criterionValue(const Plato::ScalarVector& aControl,
                                                   const Plato::Solutions& aSolution,
                                                   const std::string& aName)
{
    if (mCriteriaMap.count(aName))
    {
        Criterion tCriterion = mCriteriaMap[aName];
        return tCriterion->value(aSolution, aControl, mTimeStep);
    }
    else
    {
        ANALYZE_THROWERR(std::string("Criterion with name '") + aName +
                         std::string("' was not parsed from the 'Criteria' sublist of the input."))
    }
}

template <typename PhysicsType>
Plato::ScalarVector Problem<PhysicsType>::criterionGradient(const Plato::ScalarVector& aControl,
                                                            const Plato::Solutions& aSolution,
                                                            const std::string& aName)
{
    if (mCriteriaMap.count(aName))
    {
        Criterion tCriterion = mCriteriaMap[aName];
        return criterionGradient(aControl, aSolution, tCriterion);
    }
    else
    {
        ANALYZE_THROWERR(std::string("Criterion with name '") + aName +
                         std::string("' was not parsed from the 'Criteria' sublist of the input."))
    }
}

template <typename PhysicsType>
Plato::ScalarVector Problem<PhysicsType>::criterionGradient(const Plato::ScalarVector& aControl,
                                                            const Plato::Solutions& aSolution,
                                                            Criterion aCriterion)
{
    if (aCriterion == nullptr)
    {
        ANALYZE_THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
    }

    Plato::Solutions tSolution(mPhysics);
    tSolution.set("State", mState);
    tSolution.set("StateDot", mStateDot);

    // F_{,z}
    auto t_dFdz = aCriterion->gradient_z(tSolution, aControl, mTimeStep);

    auto tLastStepIndex = mNumSteps - 1;
    for (Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--)
    {
        auto tU = Kokkos::subview(mState, tStepIndex, Kokkos::ALL());
        auto tV = Kokkos::subview(mStateDot, tStepIndex, Kokkos::ALL());

        Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjointStates, tStepIndex, Kokkos::ALL());
        Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjointStatesV, tStepIndex, Kokkos::ALL());

        // F_{,u^k}
        auto t_dFdu = aCriterion->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
        // F_{,v^k}
        auto t_dFdv = aCriterion->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);

        if (tStepIndex != tLastStepIndex)
        {  // the last step doesn't have a contribution from k+1

            // L_{v}^{k+1}
            Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjointStatesV, tStepIndex + 1, Kokkos::ALL());

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
        const auto tJacobianU = mPDE->gradient_u_T(tU, tV, aControl, mTimeStep);

        // R_{u,v^k}
        const auto tJacobianV = mPDE->gradient_v_T(tU, tV, aControl, mTimeStep);

        // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
        Plato::blas1::axpy(-tR_vu, tJacobianV->entries(), tJacobianU->entries());

        this->applyStateConstraints(tJacobianU, t_dFdu, /*scale_constraints_by*/ 0.0);

        // L_u^k
        mSolver->solve(*tJacobianU, tAdjoint_U, t_dFdu);

        // L_v^k
        Plato::MatrixTimesVectorPlusVector(tJacobianV, tAdjoint_U, t_dFdv);
        Plato::blas1::fill(0.0, tAdjoint_V);
        Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

        // R^k_{,z}
        auto t_dRdz = mPDE->gradient_z(tU, tV, aControl, mTimeStep);

        // F_{,z} += L_u^k R^k_{,z}
        Plato::MatrixTimesVectorPlusVector(t_dRdz, tAdjoint_U, t_dFdz);
    }

    return t_dFdz;
}

template <typename PhysicsType>
Plato::ScalarVector Problem<PhysicsType>::criterionGradientX(const Plato::ScalarVector& aControl,
                                                             const Plato::Solutions& aSolution,
                                                             const std::string& aName)
{
    if (mCriteriaMap.count(aName))
    {
        Criterion tCriterion = mCriteriaMap[aName];
        return criterionGradientX(aControl, aSolution, tCriterion);
    }
    else
    {
        ANALYZE_THROWERR(std::string("Criterion with name '") + aName +
                         std::string("' was not parsed from the 'Criteria' sublist of the input."))
    }
}

template <typename PhysicsType>
Plato::ScalarVector Problem<PhysicsType>::criterionGradientX(const Plato::ScalarVector& aControl,
                                                             const Plato::Solutions& aSolution,
                                                             Criterion aCriterion)
{
    if (aCriterion == nullptr)
    {
        ANALYZE_THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
    }

    Plato::Solutions tSolution(mPhysics);
    tSolution.set("State", mState);
    tSolution.set("StateDot", mStateDot);

    // F_{,x}
    auto t_dFdx = aCriterion->gradient_x(tSolution, aControl, mTimeStep);

    auto tLastStepIndex = mNumSteps - 1;
    for (Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--)
    {
        auto tU = Kokkos::subview(mState, tStepIndex, Kokkos::ALL());
        auto tV = Kokkos::subview(mStateDot, tStepIndex, Kokkos::ALL());

        Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjointStates, tStepIndex, Kokkos::ALL());
        Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjointStatesV, tStepIndex, Kokkos::ALL());

        // F_{,u^k}
        auto t_dFdu = aCriterion->gradient_u(tSolution, aControl, tStepIndex, mTimeStep);
        // F_{,v^k}
        auto t_dFdv = aCriterion->gradient_v(tSolution, aControl, tStepIndex, mTimeStep);

        if (tStepIndex != tLastStepIndex)
        {  // the last step doesn't have a contribution from k+1

            // L_{v}^{k+1}
            Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjointStatesV, tStepIndex + 1, Kokkos::ALL());

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
        const auto tJacobianU = mPDE->gradient_u_T(tU, tV, aControl, mTimeStep);

        // R_{u,v^k}
        const auto tJacobianV = mPDE->gradient_v_T(tU, tV, aControl, mTimeStep);

        // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
        Plato::blas1::axpy(-tR_vu, tJacobianV->entries(), tJacobianU->entries());

        this->applyStateConstraints(tJacobianU, t_dFdu, /*scale_constraints_by*/ 0.0);

        // L_u^k
        mSolver->solve(*tJacobianU, tAdjoint_U, t_dFdu);

        // L_v^k
        Plato::MatrixTimesVectorPlusVector(tJacobianV, tAdjoint_U, t_dFdv);
        Plato::blas1::fill(0.0, tAdjoint_V);
        Plato::blas1::axpy(-1.0, t_dFdv, tAdjoint_V);

        // R^k_{,x}
        auto t_dRdx = mPDE->gradient_x(tU, tV, aControl, mTimeStep);

        // F_{,x} += L_u^k R^k_{,x}
        Plato::MatrixTimesVectorPlusVector(t_dRdx, tAdjoint_U, t_dFdx);
    }

    return t_dFdx;
}

template <typename PhysicsType>
Plato::Solutions Problem<PhysicsType>::getSolution() const
{
    Plato::Solutions tSolution(mPhysics, mPDEType);
    tSolution.set("State", mState, mPDE->getDofNames());
    tSolution.set("StateDot", mStateDot, mPDE->getDofDotNames());
    return tSolution;
}
}  // namespace plato::parabolic

#endif
