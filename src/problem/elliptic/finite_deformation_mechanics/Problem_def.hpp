#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_PROBLEM_DEF_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_PROBLEM_DEF_H

#include <Teuchos_ParameterList.hpp>
#include <fstream>
#include <optional>
#include <string>
#include <type_traits>

#include "boundary_conditions/ApplyConstraints.hpp"
#include "boundary_conditions/EssentialBCs.hpp"
#include "core_types/PlatoTypes.hpp"
#include "domain/AnalyzeOutput.hpp"
#include "domain/Solutions.hpp"
#include "domain/SpatialModel.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoMathHelpers.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "parsing/TeuchosParsingUtilities.hpp"
#include "problem/PlatoAbstractProblem.hpp"
#include "problem/elliptic/ScalarFunctionBaseFactory.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "solver/nonlinear_solvers/NewtonSolver.hpp"
#include "utilities/AnalyzeMacros.hpp"
#include "utilities/ParallelComm.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
template <typename PhysicsType>
Problem<PhysicsType>::Problem(Plato::Mesh aMesh, Teuchos::ParameterList& aProblemParams, Plato::Comm::Machine aMachine)
    : Plato::AbstractProblem(aMesh, aProblemParams),
      mSpatialModel(aMesh, aProblemParams, mDataMap),
      mPDE(std::make_shared<VectorFunctionType>(
          mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint"))),
      mPDEType(aProblemParams.get<std::string>("PDE Constraint")),
      mPhysics(aProblemParams.get<std::string>("Physics")),
      mOutputFileStream(aProblemParams.isParameter("Output File") ? std::optional<std::ofstream>{std::ofstream{
                                                                        aProblemParams.get<std::string>("Output File")}}
                                                                  : std::nullopt),
      mOutputStream{mOutputFileStream.has_value() ? mOutputFileStream.value() : std::cout},
      mNumSteps(Plato::ParseTools::getSubParam<int>(aProblemParams, "Time Integration", "Number Time Steps", 1)),
      mTimeStep(Plato::ParseTools::getSubParam<Plato::Scalar>(aProblemParams, "Time Integration", "Time Step", 1.0)),
      mNumNewtonSteps(Plato::ParseTools::getSubParam<int>(aProblemParams, "Newton Iteration", "Maximum Iterations", 1)),
      mNewtonIncTol(
          Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Increment Tolerance", 0.0)),
      mNewtonResTol(
          Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Residual Tolerance", 0.0)),
      mState("State", mNumSteps + 1, mPDE->size()),
      mSaveState(aProblemParams.sublist("Elliptic").isType<Teuchos::Array<std::string>>("Plottable")),
      mIsSelfAdjoint(aProblemParams.get<bool>("Self-Adjoint", false)),
      mEssentialBCs(aProblemParams.sublist("Essential Boundary Conditions", false), aMesh)
{
    Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"));
    mSolver = tSolverFactory.create(aMesh->NumNodes(), aMachine, ElementType::mNumDofsPerNode);

    if (aProblemParams.isSublist("Criteria"))
    {
        auto tAddCriteria = [&tCriteriaMap = mCriteriaMap, &tSpatialModel = mSpatialModel, &tDataMap = mDataMap,
                             &tProblemParams = aProblemParams,
                             tCriterionBaseFactory =
                                 Plato::Elliptic::ScalarFunctionBaseFactory<PhysicsType>{}](const std::string& aName)
        {
            const auto tCriterion = tCriterionBaseFactory.create(tSpatialModel, tDataMap, tProblemParams, aName);
            if (tCriterion)
            {
                tCriteriaMap[aName] = tCriterion;
            }
        };
        utilities::for_each_sublist(aProblemParams.sublist("Criteria"), tAddCriteria);
    }
}

template <typename PhysicsType>
void Problem<PhysicsType>::updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution)
{
}

template <typename PhysicsType>
Plato::Solutions Problem<PhysicsType>::solution(const Plato::ScalarVector& aControl)
{
    const auto tNewtonSolver =
        algorithms::nonlinear_solvers::NewtonSolver{mNumNewtonSteps, mNewtonResTol, mNewtonIncTol, mSolver};

    Plato::ScalarVector tInitialState = Kokkos::subview(mState, 0, Kokkos::ALL());
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tInitialState);  // zero out initial state

    Plato::OrdinalVector tBcDofs;
    Plato::ScalarVector tBcValues;
    mEssentialBCs.get(tBcDofs, tBcValues);
    Plato::ScalarVector tBcIncrementValues("BC increment values", tBcDofs.size());
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tBcIncrementValues);
    Plato::blas1::axpy(static_cast<Plato::Scalar>(1.0 / mNumSteps), tBcValues,
                       tBcIncrementValues);  // use increment of BC values

    for (Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps + 1; tStepIndex++)
    {
        mOutputStream << "\n Step " << std::to_string(tStepIndex) << "\n";
        const Plato::Scalar tTime = mTimeStep * tStepIndex;

        Plato::ScalarVector tPreviousState = Kokkos::subview(mState, tStepIndex - 1, Kokkos::ALL());
        Plato::ScalarVector tState = Kokkos::subview(mState, tStepIndex, Kokkos::ALL());
        Plato::blas1::copy(tPreviousState, tState);  // initialize current step state with previous step state

        auto tComputeResidual = [&tPDE = mPDE, &tControl = aControl,
                                 tTime](const Plato::ScalarVector& aState) -> Plato::ScalarVector
        { return tPDE->value(aState, tControl, tTime); };

        auto tComputeJacobian = [&tPDE = mPDE, &tControl = aControl,
                                 tTime](const Plato::ScalarVector& aState) -> Teuchos::RCP<Plato::CrsMatrixType>
        { return tPDE->gradient_u(aState, tControl, tTime); };

        auto tApplyBoundaryConditions =
            [tBcDofs = tBcDofs, tBcValues = tBcIncrementValues](
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

        if (mNumNewtonSteps > 1 && tNewtonHasConverged == false)
        {
            ANALYZE_THROWERR("No convergence achieved in specified number of Newton iterations.")
            break;
        }

        if (mSaveState)
        {
            const auto tResidual = mPDE->value(tState, aControl, tTime);
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
        const auto tSolution = extractLastTimeStepSolution(aSolution);
        const Criterion tCriterion = mCriteriaMap[aName];
        return tCriterion->value(tSolution, aControl);
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
        auto tComputeCriterionDerivative = [&tCriterion](const Plato::Solutions& aSolution,
                                                         const Plato::ScalarVector& aControl) -> Plato::ScalarVector
        { return tCriterion->gradient_z(aSolution, aControl); };

        auto tComputeResidualDerivative = [&tPDE = mPDE](
                                              const Plato::ScalarVector& aState,
                                              const Plato::ScalarVector& aControl) -> Teuchos::RCP<Plato::CrsMatrixType>
        { return tPDE->gradient_z(aState, aControl); };

        return computeGradient(tCriterion, aControl, aSolution, tComputeCriterionDerivative,
                               tComputeResidualDerivative);
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
                                                             const std::string& aName)
{
    if (mCriteriaMap.count(aName))
    {
        Criterion tCriterion = mCriteriaMap[aName];
        auto tComputeCriterionDerivative = [&tCriterion](const Plato::Solutions& aSolution,
                                                         const Plato::ScalarVector& aControl) -> Plato::ScalarVector
        { return tCriterion->gradient_x(aSolution, aControl); };

        auto tComputeResidualDerivative = [&tPDE = mPDE](
                                              const Plato::ScalarVector& aState,
                                              const Plato::ScalarVector& aControl) -> Teuchos::RCP<Plato::CrsMatrixType>
        { return tPDE->gradient_x(aState, aControl); };

        return computeGradient(tCriterion, aControl, aSolution, tComputeCriterionDerivative,
                               tComputeResidualDerivative);
    }
    else
    {
        ANALYZE_THROWERR(std::string("Criterion with name '") + aName +
                         std::string("' was not parsed from the 'Criteria' sublist of the input."))
    }
}

template <typename PhysicsType>
Plato::Solutions Problem<PhysicsType>::getSolution() const
{
    Plato::Solutions tSolution(mPhysics, mPDEType);
    tSolution.set("State", mState, mPDE->getDofNames());
    return tSolution;
}

template <typename PhysicsType>
void Problem<PhysicsType>::output(const std::string& aFilepath)
{
    auto tDataMap = this->getDataMap();
    auto tSolution = this->getSolution();
    auto tSolutionOutput = mPDE->getSolutionStateOutputData(tSolution);
    Plato::universal_solution_output(aFilepath, tSolutionOutput, tDataMap, mSpatialModel.Mesh);
}

template <typename PhysicsType>
Plato::Solutions Problem<PhysicsType>::extractLastTimeStepSolution(const Plato::Solutions& aSolution)
{
    const auto tNumDofs = mPDE->size();
    Plato::ScalarMultiVector tFinalState("final state", static_cast<Plato::OrdinalType>(1), tNumDofs);
    Plato::ScalarVector tState = Kokkos::subview(tFinalState, 0, Kokkos::ALL());

    Plato::ScalarVector tInputState = Kokkos::subview(aSolution.get("State"), mNumSteps, Kokkos::ALL());
    Plato::blas1::copy(tInputState, tState);

    Plato::Solutions tSolution(mPhysics, mPDEType);
    tSolution.set("State", tFinalState, mPDE->getDofNames());
    return tSolution;
}

template <typename PhysicsType>
template <typename CriterionDerivativeFunc, typename ResidualDerivativeFunc>
Plato::ScalarVector Problem<PhysicsType>::computeGradient(const Criterion& aCriterion,
                                                          const Plato::ScalarVector& aControl,
                                                          const Plato::Solutions& aSolution,
                                                          const CriterionDerivativeFunc& aComputeCriterionDerivative,
                                                          const ResidualDerivativeFunc& aComputeResidualDerivative)
{
    static_assert(
        std::is_invocable_r_v<Plato::ScalarVector, CriterionDerivativeFunc, const Plato::Solutions&,
                              const Plato::ScalarVector&>,
        "Function object CriterionDerivativeFunc has wrong signature in call to computeGradient() of Problem class.");
    static_assert(
        std::is_invocable_r_v<Teuchos::RCP<Plato::CrsMatrixType>, ResidualDerivativeFunc, const Plato::ScalarVector&,
                              const Plato::ScalarVector&>,
        "Function object ResidualDerivativeFunc has wrong signature in call to computeGradient() of Problem class.");

    const auto tSolution = extractLastTimeStepSolution(aSolution);

    const auto tPartialCriterion_PartialArgs = aComputeCriterionDerivative(tSolution, aControl);
    if (!mIsSelfAdjoint)
    {
        const auto tPartialCriterion_PartialState = aCriterion->gradient_u(tSolution, aControl, /*aStepIndex=*/0);
        Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialCriterion_PartialState);

        auto tState = Kokkos::subview(tSolution.get("State"), 0, Kokkos::ALL());
        auto tPartialResidual_PartialState = mPDE->gradient_u_T(tState, aControl);

        Plato::OrdinalVector tBcDofs;
        Plato::ScalarVector tBcValues;
        mEssentialBCs.get(tBcDofs, tBcValues);
        Plato::Scalar tScaleToZero{0.0};
        if (tPartialResidual_PartialState->isBlockMatrix())
        {
            Plato::applyBlockConstraints<ElementType::mNumDofsPerNode>(
                tPartialResidual_PartialState, tPartialCriterion_PartialState, tBcDofs, tBcValues, tScaleToZero);
        }
        else
        {
            Plato::applyConstraints<ElementType::mNumDofsPerNode>(
                tPartialResidual_PartialState, tPartialCriterion_PartialState, tBcDofs, tBcValues, tScaleToZero);
        }

        Plato::ScalarVector tAdjointVector("adjoint vector", tState.size());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tAdjointVector);
        mSolver->solve(*tPartialResidual_PartialState, tAdjointVector, tPartialCriterion_PartialState,
                       /*isAdjointSolve=*/true);

        auto tPartialResidual_PartialArgs = aComputeResidualDerivative(tState, aControl);

        Plato::MatrixTimesVectorPlusVector(tPartialResidual_PartialArgs, tAdjointVector, tPartialCriterion_PartialArgs);
    }
    return tPartialCriterion_PartialArgs;
}

}  // namespace plato::elliptic::finite_deformation_mechanics
#endif
