/*
 * Plato_Diagnostics.hpp
 *
 *  Created on: Feb 11, 2019
 */

#pragma once

#include "core_types/FadTypes.hpp"
#include "domain/Solutions.hpp"
#include "domain/WorksetBase.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoMathHelpers.hpp"
#include "mesh/PlatoMesh.hpp"
#include "problem/elliptic/AbstractScalarFunction.hpp"
#include "problem/elliptic/ScalarFunctionBase.hpp"
#include "problem/geometric/ScalarFunctionBase.hpp"
// TODO #include "LocalVectorFunctionInc.hpp"
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include "mesh/ImplicitFunctors.hpp"
#include "solver/TimeData.hpp"

namespace Plato
{

/******************************************************************************/
/**
 * \brief Test partial derivative of criterion with respect to the controls
 * \param [in] aProblem Plato problem interface
 * \param [in] aMesh    mesh database
 * return minimum finite difference error
 **********************************************************************************/
template <class PlatoProblem>
inline Plato::Scalar test_criterion_grad_wrt_control(PlatoProblem& aProblem,
                                                     Plato::Mesh aMesh,
                                                     std::string aCriterionName,
                                                     Plato::OrdinalType aSuperscriptLowerBound = 1,
                                                     Plato::OrdinalType aSuperscriptUpperBound = 6)
{
    // Allocate Data
    const Plato::OrdinalType tNumVerts = aMesh->NumNodes();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControls);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.025, 0.05, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // Compute gradient
    auto tGlobalStates = aProblem.solution(tControls);
    auto tObjGradZ = aProblem.criterionGradient(tControls, tGlobalStates, aCriterionName);
    auto tGradientDotStep = Plato::blas1::dot(tObjGradZ, tStep);

    std::ostringstream tOutput;
    tOutput << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18)
            << "FD Approx" << std::setw(20) << "abs(Error)"
            << "\n";

    std::vector<Plato::Scalar> tFiniteDiffApproxError;
    auto tTrialControl = Plato::ScalarVector("Trial Control", tNumVerts);
    for (Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        auto tEpsilon = static_cast<Plato::Scalar>(1) / std::pow(static_cast<Plato::Scalar>(10), tIndex);

        // four point finite difference approximation
        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        tGlobalStates = aProblem.solution(tTrialControl);
        auto tValuePlus1Eps = aProblem.criterionValue(tTrialControl, tGlobalStates, aCriterionName);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        tGlobalStates = aProblem.solution(tTrialControl);
        auto tValueMinus1Eps = aProblem.criterionValue(tTrialControl, tGlobalStates, aCriterionName);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tGlobalStates = aProblem.solution(tTrialControl);
        auto tValuePlus2Eps = aProblem.criterionValue(tTrialControl, tGlobalStates, aCriterionName);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tGlobalStates = aProblem.solution(tTrialControl);
        auto tValueMinus2Eps = aProblem.criterionValue(tTrialControl, tGlobalStates, aCriterionName);

        auto tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps -
                          static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
        auto tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        auto tFiniteDiffAppx = tNumerator / tDenominator;
        auto tAppxError = fabs(tFiniteDiffAppx - tGradientDotStep);
        tFiniteDiffApproxError.push_back(tAppxError);

        tOutput << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
                << tGradientDotStep << std::setw(19) << tFiniteDiffAppx << std::setw(19) << tAppxError << "\n";
    }
    std::cout << tOutput.str().c_str();

    const auto tMinError = *std::min_element(tFiniteDiffApproxError.begin(), tFiniteDiffApproxError.end());
    return tMinError;
}
// function test_criterion_grad_wrt_control

// function test_partial_control
/******************************************************************************/
/**
 * \brief Test partial derivative with respect to the control variables
 * \param [in] aMesh mesh database
 * \param [in] aCriterion scalar function (i.e. scalar criterion) interface
 * \param [in] aSuperscriptLowerBound lower bound on the superscript used to compute the step (e.g. \f$10^{lb}/$f
 **********************************************************************************/
template <typename EvaluationType, typename ElementType>
inline void test_partial_control(Plato::Mesh aMesh,
                                 Plato::Elliptic::AbstractScalarFunction<EvaluationType>& aCriterion,
                                 Plato::OrdinalType aSuperscriptLowerBound = 1,
                                 Plato::OrdinalType aSuperscriptUpperBound = 10)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh->NumElements();
    constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
    constexpr Plato::OrdinalType tDofsPerNode = ElementType::mNumDofsPerNode;
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(aMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::blas1::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // FINITE DIFFERENCE TEST
    aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
    constexpr Plato::OrdinalType tNumControlFields = 1;
    Plato::ScalarVector tPartialZ("objective partial control", tNumVerts);
    Plato::VectorEntryOrdinal<tSpaceDim, tNumControlFields> tControlEntryOrdinal(aMesh);
    Plato::assemble_scalar_gradient_fad<tNodesPerCell>(tNumCells, tControlEntryOrdinal, tResultWS, tPartialZ);

    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18)
              << "FD Approx" << std::setw(20) << "abs(Error)"
              << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    for (Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon =
            static_cast<Plato::Scalar>(1) / std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueOne = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueTwo = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueThree = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueFour = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne -
                                   static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
                  << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_control

/******************************************************************************/
/**
 * \brief Test partial derivative with respect to the state variables
 * \param [in] aMesh mesh database
 * \param [in] aCriterion scalar function (i.e. scalar criterion) interface
 **********************************************************************************/
template <typename EvaluationType, typename ElementType>
inline void test_partial_state(Plato::Mesh aMesh, Plato::Elliptic::AbstractScalarFunction<EvaluationType>& aCriterion)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh->NumElements();
    constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
    constexpr Plato::OrdinalType tDofsPerNode = ElementType::mNumDofsPerNode;
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<ElementType> tWorksetBase(aMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::blas1::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // finite difference
    aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
    Plato::ScalarVector tPartialU("objective partial state", tTotalNumDofs);
    Plato::VectorEntryOrdinal<tSpaceDim, tDofsPerNode> tStateEntryOrdinal(aMesh);
    Plato::assemble_vector_gradient_fad<tNodesPerCell, tDofsPerNode>(tNumCells, tStateEntryOrdinal, tResultWS,
                                                                     tPartialU);

    Plato::ScalarVector tStep("step", tTotalNumDofs);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialU, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18)
              << "FD Approx" << std::setw(20) << "abs(Error)"
              << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    Plato::ScalarVector tTrialState("trial state", tTotalNumDofs);
    for (Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon =
            static_cast<Plato::Scalar>(1) / std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueOne = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueTwo = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueThree = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueFour = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne -
                                   static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
                  << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_state

template <typename EvaluationType, typename ElementType>
inline void test_partial_control(Plato::Mesh aMesh,
                                 Plato::Geometric::ScalarFunctionBase& aScalarFuncBase,
                                 Plato::OrdinalType aSuperscriptLowerBound = 1,
                                 Plato::OrdinalType aSuperscriptUpperBound = 10)
{
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh->NumElements();

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // FINITE DIFFERENCE TEST
    Plato::ScalarVector tPartialZ = aScalarFuncBase.gradient_z(tControl);

    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18)
              << "FD Approx" << std::setw(20) << "abs(Error)"
              << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    for (Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon =
            static_cast<Plato::Scalar>(1) / std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueOne = aScalarFuncBase.value(tTrialControl);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueTwo = aScalarFuncBase.value(tTrialControl);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueThree = aScalarFuncBase.value(tTrialControl);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueFour = aScalarFuncBase.value(tTrialControl);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne -
                                   static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
                  << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}

// function test_partial_control
template <typename EvaluationType, typename ElementType>
inline void test_partial_control(Plato::Mesh aMesh,
                                 Plato::Elliptic::ScalarFunctionBase& aScalarFuncBase,
                                 Plato::OrdinalType aSuperscriptLowerBound = 1,
                                 Plato::OrdinalType aSuperscriptUpperBound = 10)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh->NumElements();
    constexpr Plato::OrdinalType tDofsPerNode = ElementType::mNumDofsPerNode;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarMultiVector tStates("State", /*numSteps=*/1, tTotalNumDofs);
    auto tState = Kokkos::subview(tStates, 0, Kokkos::ALL());
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::blas1::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);

    // FINITE DIFFERENCE TEST
    Plato::Solutions tSolution;
    tSolution.set("State", tStates);
    Plato::ScalarVector tPartialZ = aScalarFuncBase.gradient_z(tSolution, tControl, 0.0);

    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18)
              << "FD Approx" << std::setw(20) << "abs(Error)"
              << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    for (Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon =
            static_cast<Plato::Scalar>(1) / std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueOne = aScalarFuncBase.value(tSolution, tTrialControl, 0.0);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueTwo = aScalarFuncBase.value(tSolution, tTrialControl, 0.0);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueThree = aScalarFuncBase.value(tSolution, tTrialControl, 0.0);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueFour = aScalarFuncBase.value(tSolution, tTrialControl, 0.0);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne -
                                   static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
                  << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_control

/******************************************************************************/
/**
 * \brief Test partial derivative with respect to the state variables
 * \param [in] aMesh mesh database
 * \param [in] aCriterion scalar function (i.e. scalar criterion) interface
 **********************************************************************************/
template <typename EvaluationType, typename ElementType>
inline void test_partial_state(Plato::Mesh aMesh, Plato::Elliptic::ScalarFunctionBase& aScalarFuncBase)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh->NumElements();
    constexpr Plato::OrdinalType tDofsPerNode = ElementType::mNumDofsPerNode;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarMultiVector tStates("State", /*numSteps=*/1, tTotalNumDofs);
    auto tState = Kokkos::subview(tStates, 0, Kokkos::ALL());
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::blas1::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);

    Plato::Solutions tSolution;
    tSolution.set("State", tStates);
    Plato::ScalarVector tPartialU = aScalarFuncBase.gradient_u(tSolution, tControl, 0.0);

    Plato::ScalarMultiVector tTrialStates("trial state", /*numSteps=*/1, tTotalNumDofs);
    auto tTrialState = Kokkos::subview(tTrialStates, 0, Kokkos::ALL());

    Plato::ScalarVector tStep("step", tTotalNumDofs);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialU, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18)
              << "FD Approx" << std::setw(20) << "abs(Error)"
              << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    for (Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon =
            static_cast<Plato::Scalar>(1) / std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update<Plato::ScalarVector>(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update<Plato::ScalarVector>(tEpsilon, tStep, 1.0, tTrialState);
        tSolution.set("State", tTrialStates);
        Plato::Scalar tObjFuncValueOne = aScalarFuncBase.value(tSolution, tControl, 0.0);

        Plato::blas1::update<Plato::ScalarVector>(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update<Plato::ScalarVector>(-tEpsilon, tStep, 1.0, tTrialState);
        tSolution.set("State", tTrialStates);
        Plato::Scalar tObjFuncValueTwo = aScalarFuncBase.value(tSolution, tControl, 0.0);

        Plato::blas1::update<Plato::ScalarVector>(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update<Plato::ScalarVector>(2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tSolution.set("State", tTrialStates);
        Plato::Scalar tObjFuncValueThree = aScalarFuncBase.value(tSolution, tControl, 0.0);

        Plato::blas1::update<Plato::ScalarVector>(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update<Plato::ScalarVector>(-2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tSolution.set("State", tTrialStates);
        Plato::Scalar tObjFuncValueFour = aScalarFuncBase.value(tSolution, tControl, 0.0);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne -
                                   static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
                  << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_state

inline Plato::ScalarVector local_workset_matrix_vector_multiply(const Plato::ScalarArray3D& aWorkset,
                                                                const Plato::ScalarVector& aVector)
{
    const Plato::OrdinalType tNumCells = aWorkset.extent(0);
    const Plato::OrdinalType tNumLocalDofsPerCell = aWorkset.extent(1);
    const Plato::OrdinalType tVectorSize = aVector.extent(0);

    Plato::ScalarVector tResult("result", tVectorSize);

    Kokkos::parallel_for(
        "matrix vector multiply", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType& aCellOrdinal) {
            Plato::OrdinalType tStartingRowIndex = aCellOrdinal * tNumLocalDofsPerCell;
            for (Plato::OrdinalType tRow = 0; tRow < tNumLocalDofsPerCell; ++tRow)
            {
                tResult(tStartingRowIndex + tRow) = 0.0;
                for (Plato::OrdinalType tColumn = 0; tColumn < tNumLocalDofsPerCell; ++tColumn)
                {
                    Plato::Scalar tValue = aWorkset(aCellOrdinal, tRow, tColumn) * aVector(tStartingRowIndex + tColumn);
                    tResult(tStartingRowIndex + tRow) += tValue;
                }
            }
        });
    return tResult;
}

template <Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode>
inline Plato::ScalarVector global_workset_matrix_vector_multiply(
    const Plato::ScalarArray3D& aWorkset,
    const Plato::ScalarVector& aVector,
    const Plato::VectorEntryOrdinal<SpaceDim, DofsPerNode>& aEntryOrdinal,
    const Plato::OrdinalType& aNumNodesPerCell,
    const Plato::OrdinalType& aNumMatrixRows)
{
    const Plato::OrdinalType tNumWorksetRows = aWorkset.extent(0);
    const Plato::OrdinalType tNumWorksetCols = aWorkset.extent(1);
    const Plato::OrdinalType tVectorSize = aVector.extent(0);

    Plato::ScalarVector tResult("result", aNumMatrixRows);

    Kokkos::parallel_for(
        "matrix vector multiply", Kokkos::RangePolicy<>(0, tNumWorksetRows),
        KOKKOS_LAMBDA(const Plato::OrdinalType& aCellOrdinal) {
            for (Plato::OrdinalType tWorksetCol = 0; tWorksetCol < tNumWorksetCols; ++tWorksetCol)
            {
                Plato::OrdinalType tMatrixRow = aCellOrdinal * tNumWorksetCols + tWorksetCol;
                tResult(tMatrixRow) = 0.0;

                Plato::OrdinalType tADVarIndex = 0;
                for (Plato::OrdinalType tNode = 0; tNode < aNumNodesPerCell; ++tNode)
                {
                    for (Plato::OrdinalType tDof = 0; tDof < DofsPerNode; ++tDof)
                    {
                        Plato::OrdinalType tMatrixCol = aEntryOrdinal(aCellOrdinal, tNode, tDof);
                        Plato::Scalar tValue = aWorkset(aCellOrdinal, tWorksetCol, tADVarIndex) * aVector(tMatrixCol);
                        tResult(tMatrixRow) += tValue;
                        ++tADVarIndex;
                    }
                }
            }
        });
    return tResult;
}

template <Plato::OrdinalType D1, Plato::OrdinalType D2>
inline Plato::ScalarVector control_workset_matrix_vector_multiply(
    const Plato::ScalarArray3D& aWorkset,
    const Plato::ScalarVector& aVector,
    const Plato::VectorEntryOrdinal<D1, D2>& aEntryOrdinal,
    const Plato::OrdinalType& aNumMatrixRows)
{
    const Plato::OrdinalType tNumWorksetRows = aWorkset.extent(0);
    const Plato::OrdinalType tNumWorksetCols = aWorkset.extent(1);
    const Plato::OrdinalType tNumADvarsPerCell = aWorkset.extent(2);
    const Plato::OrdinalType tVectorSize = aVector.extent(0);

    Plato::ScalarVector tResult("result", aNumMatrixRows);

    Kokkos::parallel_for(
        "matrix vector multiply", Kokkos::RangePolicy<>(0, tNumWorksetRows),
        KOKKOS_LAMBDA(const Plato::OrdinalType& aCellOrdinal) {
            for (Plato::OrdinalType tWorksetCol = 0; tWorksetCol < tNumWorksetCols; ++tWorksetCol)
            {
                Plato::OrdinalType tMatrixRow = aCellOrdinal * tNumWorksetCols + tWorksetCol;
                tResult(tMatrixRow) = 0.0;
                for (Plato::OrdinalType tADVariableIndex = 0; tADVariableIndex < tNumADvarsPerCell; ++tADVariableIndex)
                {
                    Plato::OrdinalType tMatrixCol = aEntryOrdinal(aCellOrdinal, tADVariableIndex);
                    Plato::Scalar tValue = aWorkset(aCellOrdinal, tWorksetCol, tADVariableIndex) * aVector(tMatrixCol);
                    tResult(tMatrixRow) += tValue;
                }
            }
        });
    return tResult;
}

}  // namespace Plato
