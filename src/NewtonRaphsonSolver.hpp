/*
 * NewtonRaphsonSolver.hpp
 *
 *  Created on: Mar 2, 2020
 */

#pragma once

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "BLAS3.hpp"
#include "ParseTools.hpp"
#include "Plato_Solve.hpp"
#include "SpatialModel.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoUtilities.hpp"
#include "ApplyConstraints.hpp"
#include "alg/PlatoAbstractSolver.hpp"
#include "NewtonRaphsonUtilities.hpp"
#include "LocalVectorFunctionInc.hpp"
#include "GlobalVectorFunctionInc.hpp"
#include "InfinitesimalStrainPlasticity.hpp"
#include "InfinitesimalStrainThermoPlasticity.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Newton-Raphson solver interface.  This interface is responsible for
 * calling the Newton-Raphson solver and providing a new state.  For instance,
 * for infinitesimal strain plasticity problems, it updates the set of global
 * and local states.
 *
 * \tparam PhysicsT physics type, e.g. Plato::InfinitesimalStrainPlasticity
 *
*******************************************************************************/
template<typename PhysicsT>
class NewtonRaphsonSolver
{
// private member data
private:
    static constexpr auto mNumSpatialDims = PhysicsT::mNumSpatialDims;           /*!< spatial dimensions*/
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::mNumDofsPerCell;     /*!< number of global degrees of freedom per node*/
    static constexpr auto mNumGlobalDofsPerNode = PhysicsT::mNumDofsPerNode;     /*!< number of global degrees of freedom per node*/
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell; /*!< number of local degrees of freedom per cell (i.e. element)*/

    using LocalPhysicsT = typename PhysicsT::LocalPhysicsT;
    std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>> mGlobalEquation;    /*!< global state residual interface */
    std::shared_ptr<Plato::LocalVectorFunctionInc<LocalPhysicsT>> mLocalEquation; /*!< local state residual interface*/
    Plato::WorksetBase<PhysicsT> mWorksetBase;   /*!< interface for assembly routines */

    Plato::Scalar mStoppingTolerance;            /*!< stopping tolerance */
    Plato::Scalar mCurrentResidualNormTolerance; /*!< current residual norm stopping tolerance - avoids unnecessary solves */

    Plato::OrdinalType mMaxNumSolverIter;  /*!< maximum number of iterations */
    Plato::OrdinalType mCurrentSolverIter; /*!< current number of iterations */

    Plato::ScalarVector mDirichletValues;     /*!< Dirichlet boundary conditions values */
    Plato::OrdinalVector mDirichletDofs; /*!< Dirichlet boundary conditions degrees of freedom */

    bool mDebugFlag;              /*!< debug problem flag */
    bool mUseAbsoluteTolerance;   /*!< use absolute stopping tolerance flag */
    bool mWriteSolverDiagnostics; /*!< write solver diagnostics flag */
    std::ofstream mSolverDiagnosticsFile; /*!< output solver diagnostics */
    Plato::NewtonRaphson::measure_t mStopMeasure; /*!< solver stopping criterion measure */

    std::shared_ptr<Plato::AbstractSolver> mLinearSolver; /*!< linear solver object */

// private functions
private:
    /***************************************************************************//**
     * \brief Open Newton-Raphson solver diagnostics file.
    *******************************************************************************/
    void openDiagnosticsFile()
    {
        if (mWriteSolverDiagnostics == false)
        {
            return;
        }

        mSolverDiagnosticsFile.open("plato_analyze_newton_raphson_diagnostics.txt");
    }

    /***************************************************************************//**
     * \brief Close Newton-Raphson solver diagnostics file.
    *******************************************************************************/
    void closeDiagnosticsFile()
    {
        if (mWriteSolverDiagnostics == false)
        {
            return;
        }

        mSolverDiagnosticsFile.close();
    }

    /***************************************************************************//**
     * \brief Update the inverse of the local Jacobian with respect to the
     * current local states.
     * \param [in]     aControls set of control variables
     * \param [in]     aStates   C++ structure holding the current state variables
     * \param [in/out] Output    inverse Jacobian
    *******************************************************************************/
    void updateInverseLocalJacobian(const Plato::ScalarVector & aControls,
                                    const Plato::CurrentStates & aStates,
                                    Plato::ScalarArray3D& Output)
    {
        auto tNumCells = mLocalEquation->numCells();
        auto tDhDc = mLocalEquation->gradient_c(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                aStates.mCurrentLocalState , aStates.mPreviousLocalState,
                                                aControls, *(aStates.mTimeData));
        Plato::blas3::inverse<mNumLocalDofsPerCell, mNumLocalDofsPerCell>(tNumCells, tDhDc, Output);
    }

    /***************************************************************************//**
     * \brief Apply Dirichlet constraints to system of equations
     * \param [in/out] aMatrix   right hand side matrix
     * \param [in/out] aResidual left hand side vector
    *******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aResidual)
    {
        if(mDirichletValues.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("Newton-Raphson Solver: Essential Boundary Conditions are empty.")
        }

        Plato::ScalarVector tDispControlledDirichletValues("Dirichlet Values", mDirichletValues.size());
        Plato::blas1::fill(0.0, tDispControlledDirichletValues);
        if(mCurrentSolverIter == static_cast<Plato::OrdinalType>(0))
        {
            Plato::blas1::update(static_cast<Plato::Scalar>(1.), mDirichletValues, static_cast<Plato::Scalar>(0.), tDispControlledDirichletValues);
        }

        if(mDebugFlag == true)
        {
            // Only used for debugging purposes
            printf("Newton Raphson Solver: Apply Constraints\n");
            Plato::print_array_ordinals_1D(mDirichletDofs, "Dirichlet Dofs");
            Plato::print(mDirichletValues, "Dirichlet Values");
            Plato::print(tDispControlledDirichletValues, "Disp Controlled Dirichlet Values");
        }

        if(aMatrix->isBlockMatrix())
        {
            Plato::applyBlockConstraints<mNumGlobalDofsPerNode>(aMatrix, aResidual, mDirichletDofs, tDispControlledDirichletValues);
        }
        else
        {
            Plato::applyConstraints<mNumGlobalDofsPerNode>(aMatrix, aResidual, mDirichletDofs, tDispControlledDirichletValues);
        }
    }

    /***************************************************************************//**
     * \brief Update global states, i.e.
     *
     *   1. Solve for \f$ \delta{u} \f$, and
     *   2. Update global states, \f$ u_{i+1} = u_{i} + \delta{u} \f$,
     *
     * \param [in]     aMatrix   right hand side matrix
     * \param [in]     aResidual left hand side vector
     * \param [in/out] aStates   C++ structure with the most recent set of state variables
    *******************************************************************************/
    void updateGlobalStates(Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                            Plato::ScalarVector aResidual,
                            Plato::CurrentStates &aStates)
    {
        const Plato::Scalar tAlpha = 1.0;
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), aStates.mDeltaGlobalState);
        if (mLinearSolver == nullptr)
            ANALYZE_THROWERR("Linear solver object not initialized.")
        mLinearSolver->solve(*aMatrix, aStates.mDeltaGlobalState, aResidual);

        if(mDebugFlag == true)
        {
            //std::string tFilename = std::string("matrix_newton_iteration_") + std::to_string(mCurrentSolverIter) + ".txt";
            //Plato::print_sparse_matrix_to_file(aMatrix, tFilename);
            Plato::print(aResidual, "Residual");
            Plato::print(aStates.mDeltaGlobalState, "Delta State");
            Plato::print(aStates.mCurrentGlobalState, "Current Global State - Before Update");
        }

        Plato::blas1::update(tAlpha, aStates.mDeltaGlobalState, tAlpha, aStates.mCurrentGlobalState);

        if(mDebugFlag == true)
        {
            Plato::print(aStates.mCurrentGlobalState, "Current Global State - After Update");
        }
    }

    /***************************************************************************//**
     * \brief Compute Schur complement, i.e.
     *
     * \f$ \frac{\partial{R}}{\partial{c}} * \frac{\partial{H}}{\partial{c}}^{-1} *
     *     \frac{\partial{H}}{\partial{u}} \f$,
     *
     * where \f$ R \f$ is the global residual, \f$ H \f$ is the local residual,
     * \f$ u \f$ are the global states, and \f$ c \f$ are the local state variables.
     *
     * \param [in] aControls         current set of design variables
     * \param [in] aStates           C++ structure with the most recent set of state variables
     * \param [in] aInvLocalJacobian inverse of local Jacobian wrt local states
     *
     * \return Schur complement for each cell/element
    *******************************************************************************/
    Plato::ScalarArray3D computeSchurComplement(const Plato::ScalarVector & aControls,
                                                const CurrentStates & aStates,
                                                const Plato::ScalarArray3D & aInvLocalJacobian)
    {
        // Compute cell Jacobian of the local residual with respect to the current global state WorkSet (WS)
        auto tDhDu = mLocalEquation->gradient_u(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                 aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                 aControls, *(aStates.mTimeData));

        // Compute cell C = (dH/dc)^{-1}*dH/du, where H is the local residual, c are the local states and u are the global states
        Plato::Scalar tBeta = 0.0;
        const Plato::Scalar tAlpha = 1.0;
        auto tNumCells = mLocalEquation->numCells();
        Plato::ScalarArray3D tInvDhDcTimesDhDu("InvDhDc times DhDu", tNumCells, mNumLocalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::blas3::multiply(tNumCells, tAlpha, aInvLocalJacobian, tDhDu, tBeta, tInvDhDcTimesDhDu);

        // Compute cell Jacobian of the global residual with respect to the current local state WorkSet (WS)
        auto tDrDc = mGlobalEquation->gradient_c(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                  aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                  aStates.mProjectedPressGrad, aControls, *(aStates.mTimeData));

        // Compute cell Schur = dR/dc * (dH/dc)^{-1} * dH/du, where H is the local residual,
        // R is the global residual, c are the local states and u are the global states
        Plato::ScalarArray3D tSchurComplement("Schur Complement", tNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::blas3::multiply(tNumCells, tAlpha, tDrDc, tInvDhDcTimesDhDu, tBeta, tSchurComplement);

        return tSchurComplement;
    }

    /***************************************************************************//**
     * \brief Assemble tangent matrix, i.e.
     *
     * \f$ \frac{\partial{R}}{\partial{u}} - \left( \frac{\partial{R}}{\partial{c}}
     *   * \frac{\partial{H}}{\partial{c}}^{-1} * \frac{\partial{H}}{\partial{u}} \right) \f$
     *
     * where \f$ R \f$ is the global residual, \f$ H \f$ is the local residual,
     * \f$ u \f$ are the global states, and \f$ c \f$ are the local state variables.
     *
     * \param [in] aControls         current set of design variables
     * \param [in] aStates           C++ structure with the most recent set of state variables
     * \param [in] aInvLocalJacobian inverse of local Jacobian wrt local states
     *
     * \return Assembled tangent matrix
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    assembleTangentMatrix(const Plato::ScalarVector & aControls,
                          const Plato::CurrentStates & aStates,
                          const Plato::ScalarArray3D& aInvLocalJacobianT)
    {
        // Compute cell Schur Complement, i.e. dR/dc * (dH/dc)^{-1} * dH/du, where H is the local
        // residual, R is the global residual, c are the local states and u are the global states
        auto tSchurComplement = this->computeSchurComplement(aControls, aStates, aInvLocalJacobianT);

        // Compute cell Jacobian of the global residual with respect to the current global state WorkSet (WS)
        auto tDrDu = mGlobalEquation->gradient_u(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                   aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                   aStates.mProjectedPressGrad, aControls, *(aStates.mTimeData));

        // Add cell Schur complement to dR/du, where R is the global residual and u are the global states
        const Plato::Scalar tBeta = 1.0;
        const Plato::Scalar tAlpha = -1.0;
        auto tNumCells = mGlobalEquation->numCells();
        Plato::blas3::update(tNumCells, tAlpha, tSchurComplement, tBeta, tDrDu);

        // Assemble full Jacobian
        auto tSpatialModel = mGlobalEquation->getSpatialModel();
        auto tMesh = tSpatialModel.Mesh;
        auto tGlobalJacobian = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumGlobalDofsPerNode, mNumGlobalDofsPerNode>(tSpatialModel);
        Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumGlobalDofsPerNode> tGlobalJacEntryOrdinal(tGlobalJacobian, tMesh);
        auto tJacEntries = tGlobalJacobian->entries();
        Plato::assemble_jacobian(tNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell, tGlobalJacEntryOrdinal, tDrDu, tJacEntries);

        return tGlobalJacobian;
    }

    /***************************************************************************//**
     * \brief Assemble residual vector, i.e.
     *
     * \f$ R - \left( \frac{\partial{R}}{\partial{c}} * \frac{\partial{H}}{\partial{c}}^{-1} * H \right) \f$,
     *
     * where \f$ R \f$ is the global residual, \f$ H \f$ is the local residual, and
     * \f$ c \f$ are the local state variables.
     *
     * \param [in] aControls         current set of design variables
     * \param [in] aStates           C++ structure with the most recent set of state variables
     * \param [in] aInvLocalJacobian inverse of local Jacobian wrt local states
     *
     * \return Assembled tangent matrix
    *******************************************************************************/
    Plato::ScalarVector assembleResidual(const Plato::ScalarVector & aControls,
                                         const Plato::CurrentStates & aStates,
                                         const Plato::ScalarArray3D& aInvLocalJacobianT)
    {
        auto tGlobalResidual =
            mGlobalEquation->value(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                     aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                     aStates.mProjectedPressGrad, aControls, *(aStates.mTimeData));

        // compute local residual workset (WS)
        auto tLocalResidualWS =
                mLocalEquation->valueWorkSet(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                               aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                               aControls, *(aStates.mTimeData));

        // compute inv(DhDc)*h, where h is the local residual and DhDc is the local jacobian
        auto tNumCells = mLocalEquation->numCells();
        const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 0.0;
        Plato::ScalarMultiVector tInvLocalJacTimesLocalRes("InvLocalJacTimesLocalRes", tNumCells, mNumLocalDofsPerCell);
        Plato::blas2::matrix_times_vector("N", tAlpha, aInvLocalJacobianT, tLocalResidualWS, tBeta, tInvLocalJacTimesLocalRes);

        // compute DrDc*inv(DhDc)*h
        Plato::ScalarMultiVector tLocalResidualTerm("LocalResidualTerm", tNumCells, mNumGlobalDofsPerCell);
        auto tDrDc = mGlobalEquation->gradient_c(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                   aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                   aStates.mProjectedPressGrad, aControls, *(aStates.mTimeData));
        Plato::blas2::matrix_times_vector("N", tAlpha, tDrDc, tInvLocalJacTimesLocalRes, tBeta, tLocalResidualTerm);

        // assemble local residual contribution
        const auto tNumNodes = mGlobalEquation->numNodes();
        const auto tTotalNumDofs = mNumGlobalDofsPerNode * tNumNodes;
        Plato::ScalarVector  tLocalResidualContribution("Assembled Local Residual", tTotalNumDofs);
        mWorksetBase.assembleResidual(tLocalResidualTerm, tLocalResidualContribution);

        // add local residual contribution to global residual, i.e. r - DrDc*inv(DhDc)*h
        Plato::blas1::axpy(static_cast<Plato::Scalar>(-1.0), tLocalResidualContribution, tGlobalResidual);

        return (tGlobalResidual);
    }

    /***************************************************************************//**
      * \brief Initialize Newton-Raphson solver
      * \param [in\out] aStates C++ structure with the most recent set of state variables
     *******************************************************************************/
    void initializeSolver(Plato::CurrentStates & aStates)
    {
        mCurrentSolverIter = 0;
        Plato::blas1::update(1.0, aStates.mPreviousLocalState, 0.0, aStates.mCurrentLocalState);
        Plato::blas1::update(1.0, aStates.mPreviousGlobalState, 0.0, aStates.mCurrentGlobalState);
    }

    /***************************************************************************//**
      * \brief Check if Newton-Raphson solver converged.
      * \param [in\out] aOutputData C++ structure with the solver output diagnostics
     *******************************************************************************/
    bool didNewtonRaphsonSolverConverge(Plato::NewtonRaphsonOutputData & aOutputData)
    {
        bool tConverged = false;
        switch(aOutputData.mStopingCriterion)
        {
            case Plato::NewtonRaphson::NORM_MEASURE_TOLERANCE:
            case Plato::NewtonRaphson::CURRENT_NORM_TOLERANCE:
            {
                tConverged = true;
                break;
            }
            case Plato::NewtonRaphson::MAX_NUMBER_ITERATIONS:
            case Plato::NewtonRaphson::INFINITE_NORM_VALUE:
            case Plato::NewtonRaphson::NaN_NORM_VALUE:
            default:
            {
                tConverged = false;
                break;
            }
        }
        return (tConverged);
    }

    /***************************************************************************//**
      * \brief Check Newton-Raphson solver stopping criteria
      * \param [in\out] aOutputData C++ structure with the solver output diagnostics
     *******************************************************************************/
    bool checkStoppingCriterion(Plato::NewtonRaphsonOutputData & aOutputData)
    {
        bool tStop = false;

        if(aOutputData.mNormMeasure < mStoppingTolerance)
        {
            tStop = true;
            aOutputData.mStopingCriterion = Plato::NewtonRaphson::NORM_MEASURE_TOLERANCE;
        }
        else if(aOutputData.mCurrentNorm < mCurrentResidualNormTolerance)
        {
            tStop = true;
            aOutputData.mStopingCriterion = Plato::NewtonRaphson::CURRENT_NORM_TOLERANCE;
        }
        else if(aOutputData.mCurrentNorm >= static_cast<Plato::Scalar>(1e20))
        {
            tStop = true;
            aOutputData.mStopingCriterion = Plato::NewtonRaphson::INFINITE_NORM_VALUE;
        }
        else if(aOutputData.mCurrentIteration >= mMaxNumSolverIter)
        {
            tStop = true;
            aOutputData.mStopingCriterion = Plato::NewtonRaphson::MAX_NUMBER_ITERATIONS;
        }
        else if(!std::isfinite(aOutputData.mCurrentNorm) || !std::isfinite(aOutputData.mNormMeasure))
        {
            tStop = true;
            aOutputData.mStopingCriterion = Plato::NewtonRaphson::NaN_NORM_VALUE;
        }

        return (tStop);
    }

    /***************************************************************************//**
     * \brief Evaluate solver's stopping measure
     * \param [in] aOutputData     c++ struct with solver's diagnostics
    *******************************************************************************/
    void computeStoppingCriterion(Plato::NewtonRaphsonOutputData& aOutputData)
    {
        switch(aOutputData.mStoppingMeasure)
        {
            case Plato::NewtonRaphson::ABSOLUTE_RESIDUAL_NORM:
            {
                Plato::compute_absolute_residual_norm_error(aOutputData);
                break;
            }
            case Plato::NewtonRaphson::RELATIVE_RESIDUAL_NORM:
            {
                Plato::compute_relative_residual_norm_error(aOutputData);
                break;
            }
            default:
            {
                Plato::compute_absolute_residual_norm_error(aOutputData);
                break;
            }
        }
    }

    /***************************************************************************//**
     * \brief Finalize initialization of member data
     * \param [in] aInputs input parameters database
    *******************************************************************************/
    void initialize(Teuchos::ParameterList& aInputs)
    {
        this->openDiagnosticsFile();
        auto tStopMeasure = Plato::ParseTools::getSubParam<std::string>(aInputs, "Newton-Raphson", "Stopping Measure", "absolute residual norm");
        mStopMeasure = Plato::newton_raphson_stopping_criterion(tStopMeasure);
    }

// public functions
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh   Plato abstract mesh database
     * \param [in] aInputs input parameters database
     * \param [in] aLinearSolver linear solver object
    *******************************************************************************/
    NewtonRaphsonSolver(Plato::Mesh aMesh, Teuchos::ParameterList& aInputs, std::shared_ptr<Plato::AbstractSolver> &aLinearSolver) :
        mWorksetBase(aMesh),
        mStoppingTolerance(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputs, "Newton-Raphson", "Stopping Tolerance", 1e-8)),
        mCurrentResidualNormTolerance(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputs, "Newton-Raphson", "Current Residual Norm Stopping Tolerance", 1e-8)),
        mMaxNumSolverIter(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputs, "Newton-Raphson", "Maximum Number Iterations", 10)),
        mCurrentSolverIter(0),
        mDebugFlag(aInputs.get<bool>("Debug",false)),
        mUseAbsoluteTolerance(false),
        mWriteSolverDiagnostics(true),
        mStopMeasure(Plato::NewtonRaphson::ABSOLUTE_RESIDUAL_NORM),
        mLinearSolver(aLinearSolver)
    {
        this->initialize(aInputs);
    }

    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh Plato abstract mesh database
    *******************************************************************************/
    explicit NewtonRaphsonSolver(Plato::Mesh aMesh) :
        mWorksetBase(aMesh),
        mStoppingTolerance(1e-6),
        mCurrentResidualNormTolerance(5e-7),
        mMaxNumSolverIter(20),
        mCurrentSolverIter(0),
        mDebugFlag(false),
        mUseAbsoluteTolerance(false),
        mWriteSolverDiagnostics(true),
        mStopMeasure(Plato::NewtonRaphson::ABSOLUTE_RESIDUAL_NORM),
        mLinearSolver(nullptr)
    {
        this->openDiagnosticsFile();
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~NewtonRaphsonSolver()
    {
        this->closeDiagnosticsFile();
    }

    /***************************************************************************//**
     * \brief Set debug flag, which activate console output of diagnostics.
     * \param [in] aInput debug flag
    *******************************************************************************/
    void debug(const bool & aInput)
    {
        mDebugFlag = aInput;
    }


    /***************************************************************************//**
     * \brief Append local system of equation interface
     * \param [in] aInput local system of equation interface
    *******************************************************************************/
    void appendLocalEquation(const std::shared_ptr<Plato::LocalVectorFunctionInc<LocalPhysicsT>> & aInput)
    {
        mLocalEquation = aInput;
    }

    /***************************************************************************//**
     * \brief Append global system of equation interface
     * \param [in] aInput global system of equation interface
    *******************************************************************************/
    void appendGlobalEquation(const std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>> & aInput)
    {
        mGlobalEquation = aInput;
    }

    /***************************************************************************//**
     * \brief Append vector of Dirichlet values
     * \param [in] aInput vector of Dirichlet values
    *******************************************************************************/
    void appendDirichletValues(const Plato::ScalarVector & aInput)
    {
        mDirichletValues = aInput;
    }

    /***************************************************************************//**
     * \brief Append vector of Dirichlet degrees of freedom
     * \param [in] aInput vector of Dirichlet degrees of freedom
    *******************************************************************************/
    void appendDirichletDofs(const Plato::OrdinalVector & aInput)
    {
        mDirichletDofs = aInput;
    }

    /***************************************************************************//**
     * \brief Append output message to Newton-Raphson solver diagnostics file.
     * \param [in] aInput output message
    *******************************************************************************/
    void appendOutputMessage(const std::stringstream & aInput)
    {
        mSolverDiagnosticsFile << aInput.str().c_str();
    }

    /***************************************************************************//**
     * \brief Call Newton-Raphson solver and find new state
     * \param [in] aControls           1-D view of controls, e.g. design variables
     * \param [in] aStateData         data manager with current and previous state data
     * \param [in] aInvLocalJacobianT 3-D container for inverse Jacobian
     * \return Indicates if the Newton-Raphson solver converged (flag)
    *******************************************************************************/
    bool solve(const Plato::ScalarVector & aControls, Plato::CurrentStates & aStates)
    {
        bool tNewtonRaphsonConverged = false;
        Plato::NewtonRaphsonOutputData tOutputData;
        tOutputData.mStoppingMeasure = mStopMeasure;
        auto tNumCells = mLocalEquation->numCells();
        Plato::ScalarArray3D tInvLocalJacobianT("Inverse Transpose DhDc", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        tOutputData.mWriteOutput = mWriteSolverDiagnostics;
        Plato::print_newton_raphson_diagnostics_header(tOutputData, mSolverDiagnosticsFile);

        this->initializeSolver(aStates);
        // Elastic trial step
        mLocalEquation->updateLocalState(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                         aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                         aControls, *(aStates.mTimeData));
        while(true)
        {
            tOutputData.mCurrentIteration = mCurrentSolverIter;
            if (mDebugFlag) printf("Iter: %d\nUpdate Local Jacobian Inverse.\n", mCurrentSolverIter);
            // update inverse of local Jacobian -> store in tInvLocalJacobianT
            this->updateInverseLocalJacobian(aControls, aStates, tInvLocalJacobianT);
            if (mDebugFlag) printf("Assemble residual.\n");
            // assemble residual
            auto tGlobalResidual = this->assembleResidual(aControls, aStates, tInvLocalJacobianT);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1.0), tGlobalResidual);
            if (mDebugFlag) printf("Assemble tangent.\n");
            // assemble tangent stiffness matrix
            auto tGlobalJacobian = this->assembleTangentMatrix(aControls, aStates, tInvLocalJacobianT);

            // apply Dirichlet boundary conditions
            this->applyConstraints(tGlobalJacobian, tGlobalResidual);

            // check convergence
            tOutputData.mCurrentNorm = Plato::blas1::norm(tGlobalResidual);
            this->computeStoppingCriterion(tOutputData);
            Plato::print_newton_raphson_diagnostics(tOutputData, mSolverDiagnosticsFile);

            const bool tStop = this->checkStoppingCriterion(tOutputData);
            if(tStop == true)
            {
                tNewtonRaphsonConverged = this->didNewtonRaphsonSolverConverge(tOutputData);
                break;
            }
            if (mDebugFlag) printf("Update global states.\n");
            // update global states
            this->updateGlobalStates(tGlobalJacobian, tGlobalResidual, aStates);
            if (mDebugFlag) printf("Update local states.\n");
            // update local states
            mLocalEquation->updateLocalState(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                             aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                             aControls, *(aStates.mTimeData));
            mCurrentSolverIter++;
        }
        if (mDebugFlag) printf("Newton iteration completed.\n");
        Plato::print_newton_raphson_stop_criterion(tOutputData, mSolverDiagnosticsFile);
        
        return (tNewtonRaphsonConverged);
    }
};
// class NewtonRaphsonSolver

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<1>>;
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainThermoPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<2>>;
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<3>>;
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif

