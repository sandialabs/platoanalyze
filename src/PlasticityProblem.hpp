/*
 * PlasticityProblem.hpp
 *
 *  Created on: Mar 2, 2020
 */

#pragma once

#include <memory>

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "Solutions.hpp"
#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "EssentialBCs.hpp"
#include "AnalyzeOutput.hpp"
#include "NewtonRaphsonSolver.hpp"
#include "PlatoAbstractProblem.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "alg/PlatoAbstractSolver.hpp"
#include "PathDependentAdjointSolver.hpp"
#include "InfinitesimalStrainPlasticity.hpp"
#include "InfinitesimalStrainThermoPlasticity.hpp"
#include "PathDependentScalarFunctionFactory.hpp"
#include "TimeData.hpp"

//temporary (until mesh IO is done)
#include "OmegaHMesh.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Plasticity problem manager.  This interface is responsible for the
 * evaluation of the criteria value, criteria sensitivities, and residual.
 *
 * \tparam PhysicsT physics type, e.g. Plato::InfinitesimalStrainPlasticity
*******************************************************************************/
template<typename PhysicsT>
class PlasticityProblem : public Plato::AbstractProblem
{

// private member data
private:

    using Criterion = std::shared_ptr<Plato::LocalScalarFunctionInc>;
    using Criteria  = std::map<std::string, Criterion>;

    static constexpr auto mSpaceDim                = PhysicsT::mSpaceDim;             /*!< spatial dimensions*/
    static constexpr auto mNumNodesPerCell         = PhysicsT::mNumNodesPerCell;      /*!< number of nodes per cell*/
    static constexpr auto mPressureDofOffset       = PhysicsT::mPressureDofOffset;    /*!< number of pressure dofs offset*/
    static constexpr auto mTemperatureDofOffset    = PhysicsT::mTemperatureDofOffset; /*!< number of temperature dofs offset*/
    static constexpr auto mNumGlobalDofsPerNode    = PhysicsT::mNumDofsPerNode;       /*!< number of global degrees of freedom per node*/
    static constexpr auto mNumGlobalDofsPerCell    = PhysicsT::mNumDofsPerCell;       /*!< number of global degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumLocalDofsPerCell     = PhysicsT::mNumLocalDofsPerCell;  /*!< number of local degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumPressGradDofsPerCell = PhysicsT::mNumNodeStatePerCell;  /*!< number of projected pressure gradient degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumPressGradDofsPerNode = PhysicsT::mNumNodeStatePerNode;  /*!< number of projected pressure gradient degrees of freedom per node*/
    static constexpr auto mNumConfigDofsPerCell    = mSpaceDim * mNumNodesPerCell;    /*!< number of configuration (i.e. coordinates) degrees of freedom per cell (i.e. element) */

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // Required
    using PlasticityT = typename PhysicsT::LocalPhysicsT;
    using ProjectorT  = typename Plato::Projection<mSpaceDim, PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>;
    std::shared_ptr<Plato::VectorFunctionVMS<ProjectorT>> mProjectionEquation;  /*!< global pressure gradient projection interface */
    std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>> mGlobalEquation;  /*!< global equality constraint interface */
    std::shared_ptr<Plato::LocalVectorFunctionInc<PlasticityT>> mLocalEquation; /*!< local equality constraint interface */
    std::shared_ptr<Plato::TimeData> mTimeData;                                 /*!< time data object */

    // Optional
    Criteria mCriteria;

    Plato::Scalar mPressureScaling;               /*!< pressure term scaling */
    Plato::Scalar mTemperatureScaling;            /*!< temperature term scaling */
    Plato::Scalar mReferenceTemperature;          /*!< reference temperature */
    Plato::Scalar mInitialNormResidual;           /*!< initial norm of global residual */
    Plato::Scalar mDispControlConstant;           /*!< displacement control constant */

    Plato::ScalarVector mPressure;                /*!< projected pressure field */
    Plato::ScalarVector mControl;                 /*!< control variables for output */
    Plato::ScalarMultiVector mLocalStates;        /*!< local state variables */
    Plato::ScalarMultiVector mGlobalStates;       /*!< global state variables */
    Plato::ScalarMultiVector mReactionForce;      /*!< reaction */
    Plato::ScalarMultiVector mProjectedPressGrad; /*!< projected pressure gradient (# Time Steps, # Projected Pressure Gradient dofs) */

    std::shared_ptr<Plato::EssentialBCs<PhysicsT>> mEssentialBCs; /*!< essential boundary conditions shared pointer */
    Plato::ScalarVector mPreviousStepDirichletValues;            /*!< previous time step values associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mDirichletValues;         /*!< values associated with the Dirichlet boundary conditions */
    Plato::OrdinalVector mDirichletDofs;     /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */

    Plato::WorksetBase<PhysicsT> mWorksetBase;    /*!< assembly routine interface */

    Plato::SolverFactory mLinearSolverFactory;            /*!< linear solver factory */
    std::shared_ptr<Plato::AbstractSolver> mLinearSolver; /*!< linear solver object */

    std::shared_ptr<Plato::NewtonRaphsonSolver<PhysicsT>> mNewtonSolver;         /*!< Newton-Raphson solve interface */
    std::shared_ptr<Plato::PathDependentAdjointSolver<PhysicsT>> mAdjointSolver; /*!< Path-dependent adjoint solver interface */

    bool mStopOptimization; /*!< stops optimization problem if Newton-Raphson solver fails to converge during an optimization run */

    std::string mPDEType; /*!< partial differential equation type */
    std::string mPhysics; /*!< simulated physics */

// public functions
public:
    /***************************************************************************//**
     * \brief Plasticity problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aInputs input parameters database
     * \param [in] aMachine MPI communicator wrapper
    *******************************************************************************/
    PlasticityProblem(
      Plato::Mesh              aMesh,
      Teuchos::ParameterList & aInputs,
      Comm::Machine          & aMachine
    ) :
      mSpatialModel(aMesh, aInputs),
      mLocalEquation(std::make_shared<Plato::LocalVectorFunctionInc<PlasticityT>>(mSpatialModel, mDataMap, aInputs)),
      mGlobalEquation(std::make_shared<Plato::GlobalVectorFunctionInc<PhysicsT>>(mSpatialModel, mDataMap, aInputs, aInputs.get<std::string>("PDE Constraint"))),
      mProjectionEquation(std::make_shared<Plato::VectorFunctionVMS<ProjectorT>>(mSpatialModel, mDataMap, aInputs, std::string("State Gradient Projection"))),
      mTimeData(std::make_shared<Plato::TimeData>(aInputs)),
      mPressureScaling(1.0),
      mTemperatureScaling(1.0),
      mReferenceTemperature(1.0),
      mInitialNormResidual(std::numeric_limits<Plato::Scalar>::max()),
      mDispControlConstant(std::numeric_limits<Plato::Scalar>::min()),
      mPressure("Previous Pressure Field", aMesh->NumNodes()),
      mLocalStates("Local States", mTimeData->mNumTimeSteps, mLocalEquation->size()),
      mGlobalStates("Global States", mTimeData->mNumTimeSteps, mGlobalEquation->size()),
      mReactionForce("Reaction Force", mTimeData->mNumTimeSteps, aMesh->NumNodes()),
      mProjectedPressGrad("Projected Pressure Gradient", mTimeData->mNumTimeSteps, mProjectionEquation->size()),
      mWorksetBase(aMesh),
      mLinearSolverFactory(aInputs.sublist("Linear Solver")),
      mLinearSolver(mLinearSolverFactory.create(aMesh->NumNodes(), aMachine, PhysicsT::mNumDofsPerNode)),
      mNewtonSolver(std::make_shared<Plato::NewtonRaphsonSolver<PhysicsT>>(aMesh, aInputs, mLinearSolver)),
      mAdjointSolver(std::make_shared<Plato::PathDependentAdjointSolver<PhysicsT>>(aMesh, aInputs, mLinearSolver)),
      mStopOptimization(false),
      mPDEType(aInputs.get<std::string>("PDE Constraint")),
      mPhysics(aInputs.get<std::string>("Physics")),
      mEssentialBCs(nullptr)
    {
        this->initialize(aInputs);
    }

    /***************************************************************************//**
     * \brief Append global residual evaluation interface
     * \param [in] aInput global residual evaluation interface
    *******************************************************************************/
    void appendGlobalResidual(const std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>>& aInput)
    {
        mGlobalEquation = aInput;
    }

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aInputs input parameters database
    *******************************************************************************/
    void readEssentialBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Essential Boundary Conditions") == false)
        {
            ANALYZE_THROWERR("Plasticity Problem: Essential Boundary Conditions are not defined for this problem.")
        }
        Plato::OrdinalType tPressureDofOffset    = mPressureDofOffset;
        Plato::OrdinalType tTemperatureDofOffset = mTemperatureDofOffset;

        std::map<Plato::OrdinalType, Plato::Scalar> tDofOffsetToScaleFactor;
        tDofOffsetToScaleFactor[tPressureDofOffset]    = mPressureScaling;
        tDofOffsetToScaleFactor[tTemperatureDofOffset] = mTemperatureScaling;

        mEssentialBCs =
        std::make_shared<Plato::EssentialBCs<PhysicsT>>
             (aInputs.sublist("Essential Boundary Conditions", false), mSpatialModel.Mesh, tDofOffsetToScaleFactor);
        mEssentialBCs->get(mDirichletDofs, mDirichletValues); // BCs at time = 0
        Kokkos::resize(mPreviousStepDirichletValues, mDirichletValues.size());
        Plato::blas1::fill(0.0, mPreviousStepDirichletValues);
    }

    /***************************************************************************//**
     * \brief Set Dirichlet boundary conditions
     * \param [in] aDirichletDofs   degrees of freedom associated with Dirichlet boundary conditions
     * \param [in] aDirichletValues values associated with Dirichlet degrees of freedom
    *******************************************************************************/
    void setEssentialBoundaryConditions(const Plato::OrdinalVector & aDirichletDofs, const Plato::ScalarVector & aDirichletValues)
    {
        if(aDirichletDofs.size() != aDirichletValues.size())
        {
            std::ostringstream tError;
            tError << "PLASTICITY PROBLEM: DIMENSION MISMATCH: THE NUMBER OF ELEMENTS IN INPUT DOFS AND VALUES ARRAY DO NOT MATCH."
                << "DOFS SIZE = " << aDirichletDofs.size() << " AND VALUES SIZE = " << aDirichletValues.size();
            ANALYZE_THROWERR(tError.str())
        }
        mDirichletDofs   = aDirichletDofs;
        mDirichletValues = aDirichletValues;
        Kokkos::resize(mPreviousStepDirichletValues, aDirichletValues.size());
        Plato::blas1::fill(0.0, mPreviousStepDirichletValues);
    }

    /***************************************************************************//**
     * \brief Save states to visualization file
     * \param [in] aFilepath output/viz directory path
    *******************************************************************************/
    void output(const std::string& aFilepath)
    {
        auto tNumNodes = mGlobalEquation->numNodes();
        Plato::ScalarMultiVector tPressure("pressure", mGlobalStates.extent(0), tNumNodes);
        Plato::ScalarMultiVector tTemperature("temperature", mGlobalStates.extent(0), tNumNodes);
        Plato::ScalarMultiVector tDisplacements("displacement", mGlobalStates.extent(0), tNumNodes*mSpaceDim);
        Plato::blas2::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(mGlobalStates, tPressure);
        Plato::blas2::extract<mNumGlobalDofsPerNode, mSpaceDim>(tNumNodes, mGlobalStates, tDisplacements);
        Plato::blas2::scale(mPressureScaling, tPressure);
        if (mTemperatureDofOffset > 0)
        {
          Plato::blas2::extract<mNumGlobalDofsPerNode, mTemperatureDofOffset>(mGlobalStates, tTemperature);
          Plato::blas2::scale(mTemperatureScaling, tTemperature);
        }

        auto tWriter = Plato::MeshIOFactory::create(aFilepath, mSpatialModel.Mesh, "Write");

        for(Plato::OrdinalType tSnapshot = 0; tSnapshot < tDisplacements.extent(0); tSnapshot++)
        {
            auto tPressSubView = Kokkos::subview(tPressure, tSnapshot, Kokkos::ALL());
            tWriter->AddNodeData("pressure", tPressSubView);

            auto tForceSubView = Kokkos::subview(mReactionForce, tSnapshot, Kokkos::ALL());
            tWriter->AddNodeData("reaction force", tForceSubView);

            auto tDispSubView = Kokkos::subview(tDisplacements, tSnapshot, Kokkos::ALL());
            tWriter->AddNodeData("displacement", tDispSubView, mSpaceDim);

            if (mTemperatureDofOffset > 0)
            {
              auto tTemperatureSubView = Kokkos::subview(tTemperature, tSnapshot, Kokkos::ALL());
              tWriter->AddNodeData("temperature", tTemperatureSubView);

            }

            Plato::AddStateData(tWriter, mDataMap.getState(tSnapshot), mSpaceDim);

            auto tTime = mTimeData->mCurrentTimeStepSize * static_cast<Plato::Scalar>(tSnapshot + 1);
            tWriter->Write(tSnapshot, tTime);
        }
    }

    /***************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    *******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector) {return;}

    /***************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControls 1D container of control variables
     * \param [in] aSolution solution database
    *******************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControls,
                       const Plato::Solutions    & aSolution) override
    {
        auto tGlobalState = aSolution.get("State");
        mLocalEquation->updateProblem(tGlobalState, mLocalStates, aControls, *mTimeData);
        mGlobalEquation->updateProblem(tGlobalState, mLocalStates, aControls, *mTimeData);
        mProjectionEquation->updateProblem(tGlobalState, aControls, mTimeData->mCurrentTime);

        for( auto tCriterion : mCriteria )
        {
            tCriterion.second->updateProblem(tGlobalState, mLocalStates, aControls, *mTimeData);
        }
    }

    /***************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControls 1D view of control variables
     * \return solution database
    *******************************************************************************/
    Plato::Solutions solution(const Plato::ScalarVector &aControls) override
    {
        // TODO: NOTES
        // 1. WRITE LOCAL STATES, PRESSURE, AND GLOBAL STATES HISTORY TO FILE - MEMORY CONCERNS
        //   1.1. NO NEED TO STORE MEMBER DATA FOR THESE QUANTITIES
        //   1.2. READ DATA FROM FILES DURING ADJOINT SOLVE
        // 4. HOW WILL OUTPUT DATA BE PRESENTED TO THE USERS, WE CANNOT SEND TIME-DEPENDENT DATA THROUGH THE ENGINE.
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: INPUT CONTROL VECTOR IS EMPTY.")
        }
        
        mDataMap.scalarNodeFields["Topology"] = aControls;

        bool tStop = false;
        while (tStop == false)
        {
            bool tDidSolverConverge = this->solveForwardProblem(aControls);
            if (tDidSolverConverge == true)
            {
                tStop = true;
                std::stringstream tMsg;
                tMsg << "\n**** Forward Solve Was Successful ****\n";
                mNewtonSolver->appendOutputMessage(tMsg);
                break;
            }
            else
            {
                if(mTimeData->mMaxNumTimeStepsReached == true)
                {
                    tStop = true;
                    std::stringstream tMsg;
                    tMsg << "\n**** Maximum Number of Pseudo Time Steps Was Reached. "
                            << "Plasticity Problem failed to converge to a solution. ****\n";
                    REPORT(tMsg.str().c_str());
                    mNewtonSolver->appendOutputMessage(tMsg);
                    mStopOptimization = true;
                    break;
                }
                this->resizeTimeDependentStates();
                std::stringstream tMsg;
                tMsg << "\n**** Forward Solve Was Not Successful ****\n"
                        << "Number of pseudo time steps will be increased to '" << mTimeData->mNumTimeSteps << "'\n.";
                REPORT(tMsg.str().c_str());
                mNewtonSolver->appendOutputMessage(tMsg);
            }
        }

        Plato::Solutions tSolution(mPhysics);
        tSolution.set("State", mGlobalStates);
        return tSolution;
    }

    /***************************************************************************//**
     * \fn Plato::Scalar criterionValue
     * \brief Evaluate criterion function and return its value
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return criterion function value
    *******************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: SOLUTION DATABASE IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];

            this->shouldOptimizationProblemStop();
            auto tGlobalState = aSolution.get("State"); 
            auto tOutput = this->evaluateCriterion(*tCriterion, tGlobalState, mLocalStates, aControls);

            return (tOutput);
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            ANALYZE_THROWERR(tErrorMessage);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion function and return its value
     * \param [in] aControls 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
    *******************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControls,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];

            this->shouldOptimizationProblemStop();
            auto tOutput = this->evaluateCriterion(*tCriterion, mGlobalStates, mLocalStates, aControls);

            return (tOutput);
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            ANALYZE_THROWERR(tErrorMessage);
        }

    }

    /***************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt control variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControls,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];

            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mGlobalStates);
            this->shouldOptimizationProblemStop();
            auto tTotalDerivative = this->criterionGradient(aControls, tSolution, tCriterion);

            return tTotalDerivative;
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            ANALYZE_THROWERR(tErrorMessage);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view of the criterion gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: SOLUTION DATABASE IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];

            this->shouldOptimizationProblemStop();
            auto tTotalDerivative = this->criterionGradient(aControls, aSolution, tCriterion);

            return (tTotalDerivative);
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            ANALYZE_THROWERR(tErrorMessage);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view of the criterion gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion
    )
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: SOLUTION DATABASE IS EMPTY.");
        }

        this->shouldOptimizationProblemStop();
        mAdjointSolver->appendScalarFunction(aCriterion);

        auto tNumNodes = mGlobalEquation->numNodes();
        Plato::ScalarVector tTotalDerivative("Total Derivative", tNumNodes);
        this->backwardTimeIntegration(Plato::PartialDerivative::CONTROL, aControls, tTotalDerivative);
        this->addCriterionPartialDerivativeZ(*aCriterion, aControls, tTotalDerivative);

        return (tTotalDerivative);
    }

    /***************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt configuration variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControls,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            
            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mGlobalStates);
            this->shouldOptimizationProblemStop();
            auto tTotalDerivative = this->criterionGradientX(aControls, tSolution, tCriterion);

            return tTotalDerivative;
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            ANALYZE_THROWERR(tErrorMessage);
        }

    }

    /***************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view of the criterion gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: SOLUTIONS DATABASE IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            auto tTotalDerivative = this->criterionGradientX(aControls, aSolution, tCriterion);

            return (tTotalDerivative);
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            ANALYZE_THROWERR(tErrorMessage);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view of the criterion gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion
    )
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            ANALYZE_THROWERR("PLASTICITY PROBLEM: SOLUTION DATABASE IS EMPTY.");
        }


        this->shouldOptimizationProblemStop();
        mAdjointSolver->appendScalarFunction(aCriterion);
        Plato::ScalarVector tTotalDerivative("Total Derivative", mNumConfigDofsPerCell);
        this->backwardTimeIntegration(Plato::PartialDerivative::CONFIGURATION, aControls, tTotalDerivative);
        this->addCriterionPartialDerivativeX(*aCriterion, aControls, tTotalDerivative);

        return (tTotalDerivative);
    }

    /***************************************************************************//**
     * \brief Compute reaction force.
     * \param [in] aControl 1D view of controls
     * \param [in] aStates  C++ structure with current state information
    *******************************************************************************/
    void computeReactionForce(const Plato::ScalarVector &aControl, Plato::CurrentStates &aStates)
    {
        mDataMap.mScalarValues["LoadControlConstant"] = 0.0;
        auto tInternalForce = mGlobalEquation->value(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                     aStates.mCurrentLocalState,  aStates.mPreviousLocalState,
                                                     aStates.mProjectedPressGrad, aControl, *(aStates.mTimeData));

        auto tNumNodes = mGlobalEquation->numNodes();
        auto tReactionForce = Kokkos::subview(mReactionForce, aStates.mCurrentStepIndex, Kokkos::ALL());
        Kokkos::parallel_for("reaction force", Kokkos::RangePolicy<>(0, tNumNodes), KOKKOS_LAMBDA(const Plato::OrdinalType &aOrdinal)
        {
            for(Plato::OrdinalType tDim = 0; tDim < mSpaceDim; tDim++)
            {
                tReactionForce(aOrdinal) += tInternalForce(aOrdinal*mNumGlobalDofsPerNode+tDim);
            }
        });
    }

    /***************************************************************************//**
     * \brief Set initial temperature field equal to the reference temperature
     * \param [in]     aCurrentStepIndex    current time step index
     * \param [in/out] aPreviousGlobalState previous global state
    *******************************************************************************/
    void setInitialTemperature(const Plato::OrdinalType & aCurrentStepIndex,
                               const Plato::ScalarVector & aPreviousGlobalState) const
    {
        if (aCurrentStepIndex != static_cast<Plato::OrdinalType>(0)) return;

        auto tReferenceTemperature = mReferenceTemperature;
        auto tTemperatureScaling   = mTemperatureScaling;
        auto tNumGlobalDofsPerNode = mNumGlobalDofsPerNode;
        auto tTemperatureDofOffset = mTemperatureDofOffset;
        auto tNumVerts             = mSpatialModel.Mesh->NumNodes();

        if (tTemperatureDofOffset > static_cast<Plato::OrdinalType>(0))
        {
            Kokkos::parallel_for("set temperature to reference", Kokkos::RangePolicy<>(0, tNumVerts),
                                    KOKKOS_LAMBDA(const Plato::OrdinalType &aVertexOrdinal)
            {
                Plato::OrdinalType tIndex = aVertexOrdinal * tNumGlobalDofsPerNode + tTemperatureDofOffset;
                aPreviousGlobalState(tIndex) = tReferenceTemperature / tTemperatureScaling;
            });
        }
    }

// private functions
private:
    /***************************************************************************//**
     * \brief Initialize member data
     * \param [in] aInputParams  input parameter list
    *******************************************************************************/
    void initialize(Teuchos::ParameterList& aInputParams)
    {
        this->allocateCriteria(aInputParams);

        if(aInputParams.isSublist("Material Models") == false)
        {
            ANALYZE_THROWERR("Plasticity Problem: 'Material Models' Parameter Sublist is not defined.")
        }
        Teuchos::ParameterList tMaterialsInputs = aInputParams.get<Teuchos::ParameterList>("Material Models");

        mPressureScaling    = tMaterialsInputs.get<Plato::Scalar>("Pressure Scaling", 1.0);
        mTemperatureScaling = tMaterialsInputs.get<Plato::Scalar>("Temperature Scaling", 1.0);

        // Get the reference temperature and make sure all materials have an identical reference temperature
        if (mTemperatureDofOffset > 0)
        {
            std::vector<Plato::Scalar> tReferenceTemperatures;
            for(Teuchos::ParameterList::ConstIterator tIndex =  tMaterialsInputs.begin(); 
                                                      tIndex != tMaterialsInputs.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry & tEntry = tMaterialsInputs.entry(tIndex);
                if (!tEntry.isList()) continue;

                std::string tName = tMaterialsInputs.name(tIndex);
                Teuchos::ParameterList tMaterialList = tMaterialsInputs.sublist(tName);

                if (!tMaterialList.isSublist("Isotropic Linear Thermoelastic")) continue;
                auto tThermoelasticSubList = tMaterialList.sublist("Isotropic Linear Thermoelastic");
                auto tReferenceTemperature = tThermoelasticSubList.get<Plato::Scalar>("Reference Temperature");
                tReferenceTemperatures.push_back(tReferenceTemperature);
            }
            if (tReferenceTemperatures.empty())
            {
                ANALYZE_THROWERR("Reference temperature should be set for at least one material.")
            }
            else if (tReferenceTemperatures.size() == 1)
            {
                mReferenceTemperature = tReferenceTemperatures[0];
            }
            else
            {
                mReferenceTemperature = tReferenceTemperatures[0];
                for (Plato::OrdinalType tIndex = 0; tIndex < tReferenceTemperatures.size(); ++tIndex)
                    if (std::abs(mReferenceTemperature - tReferenceTemperatures[tIndex]) > 5.0e-7)
                        ANALYZE_THROWERR("All materials must have the same reference temperature.")
            }
        }

    }

    /***************************************************************************//**
     * \brief Checks if optimization problem should be stop due to Newton-Raphson
     * solver's failure to converge during optimization.
    *******************************************************************************/
    void shouldOptimizationProblemStop()
    {
        if(mStopOptimization == true)
        {
            std::stringstream tMsg;
            tMsg << "\n**** Plasticity Problem: Newton-Raphson solver failed to converge during optimization. "
                    << "Optimization results are going to be impacted by the solver's failure to converge. ****\n";
            ANALYZE_THROWERR(tMsg.str().c_str())
        }
    }

    /***************************************************************************//**
     * \brief Resize time-dependent state containers and decrease pseudo-time step size.
    *******************************************************************************/
    void resizeTimeDependentStates()
    {
        mTimeData->increaseNumTimeSteps();

        Plato::blas1::fill(0.0, mPreviousStepDirichletValues);

        Kokkos::resize(mLocalStates, mTimeData->mNumTimeSteps, mLocalEquation->size());
        Kokkos::resize(mGlobalStates, mTimeData->mNumTimeSteps, mGlobalEquation->size());
        Kokkos::resize(mReactionForce, mTimeData->mNumTimeSteps, mGlobalEquation->numNodes());
        Kokkos::resize(mProjectedPressGrad, mTimeData->mNumTimeSteps, mProjectionEquation->size());
    }

    /***************************************************************************//**
     * \brief Initialize Newton-Raphson solver
    *******************************************************************************/
    void initializeNewtonSolver()
    {
        mNewtonSolver->appendDirichletDofs(mDirichletDofs);
        mNewtonSolver->appendDirichletValues(mDirichletValues);

        mNewtonSolver->appendLocalEquation(mLocalEquation);
        mNewtonSolver->appendGlobalEquation(mGlobalEquation);

        std::stringstream tMsg("\n\n**** NEW NEWTON-RAPHSON SOLVE ****\n\n");
        mNewtonSolver->appendOutputMessage(tMsg);
    }

    /***************************************************************************//**
     * \brief Solve forward problem
     * \param [in] aControls 1-D view of controls, e.g. design variables
     * \return flag used to indicate forward problem was solved to completion
    *******************************************************************************/
    bool solveForwardProblem(const Plato::ScalarVector & aControls)
    {
        mDataMap.clearStates();

        Plato::CurrentStates tCurrentState(mTimeData);
        auto tNumCells = mLocalEquation->numCells();
        tCurrentState.mDeltaGlobalState = Plato::ScalarVector("Global State Increment", mGlobalEquation->size());

        this->initializeNewtonSolver();
        Plato::blas2::fill(static_cast<Plato::Scalar>(0.0), mLocalStates);
        Plato::blas2::fill(static_cast<Plato::Scalar>(0.0), mGlobalStates);
        Plato::blas2::fill(static_cast<Plato::Scalar>(0.0), mProjectedPressGrad);

        bool tForwardProblemSolved = false;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mTimeData->mNumTimeSteps; tCurrentStepIndex++)
        {
            mTimeData->updateTimeData(tCurrentStepIndex);

            std::stringstream tMsg;
            tMsg << "TIME STEP #" << mTimeData->getTimeStepIndexPlusOne() << " OUT OF " << mTimeData->mNumTimeSteps
                 << " TIME STEPS, TOTAL TIME = " << mTimeData->mCurrentTime << "\n";
            mNewtonSolver->appendOutputMessage(tMsg);

            tCurrentState.mCurrentStepIndex = tCurrentStepIndex;
            this->cacheStateData(tCurrentState);

            // update displacement and load control multiplier
            this->updateDispAndLoadControlMultipliers(tCurrentStepIndex);

            // update local and global states
            bool tNewtonRaphsonConverged = mNewtonSolver->solve(aControls, tCurrentState);
            mDataMap.saveState();

            // compute reaction force
            this->computeReactionForce(aControls, tCurrentState);

            if(tNewtonRaphsonConverged == false)
            {
                std::stringstream tMsg;
                tMsg << "**** Newton-Raphson Solver did not converge at time step #"
                     << mTimeData->getTimeStepIndexPlusOne()
                     << ".  Number of pseudo time steps will be increased to '"
                     << static_cast<Plato::OrdinalType>(mTimeData->mNumTimeSteps * mTimeData->mTimeStepExpansionMultiplier) << "'. ****\n\n";
                mNewtonSolver->appendOutputMessage(tMsg);
                return tForwardProblemSolved;
            }

            // update projected pressure gradient state
            this->updateProjectedPressureGradient(aControls, tCurrentState);
        }
        tForwardProblemSolved = true;
        // mControl = aControls;
        // this->saveStates("PlasticityOutput");
        return tForwardProblemSolved;
    }

    /***************************************************************************//**
     * \brief Update displacement and load control multiplier.
     * \param [in] aInput  current time step index
    *******************************************************************************/
    void updateDispAndLoadControlMultipliers(const Plato::OrdinalType& aInput)
    {
        auto tLoadControlConstant = mTimeData->mCurrentTimeStepSize * static_cast<Plato::Scalar>(aInput + 1);
        mDataMap.mScalarValues["LoadControlConstant"] = tLoadControlConstant;

        if (aInput != static_cast<Plato::OrdinalType>(0))
          Plato::blas1::copy(mDirichletValues, mPreviousStepDirichletValues);
        else
          Plato::blas1::fill(0.0, mPreviousStepDirichletValues);

        auto tCurrentTime = tLoadControlConstant;
        if (mEssentialBCs != nullptr)
          mEssentialBCs->get(mDirichletDofs, mDirichletValues, tCurrentTime);
        else
          ANALYZE_THROWERR("EssentialBCs pointer is null!")

        Plato::ScalarVector tNewtonUpdateDirichletValues("Dirichlet Increment Values", mDirichletValues.size());
        Plato::blas1::copy(mDirichletValues, tNewtonUpdateDirichletValues);
        Plato::blas1::axpy(static_cast<Plato::Scalar>(-1.0), mPreviousStepDirichletValues, tNewtonUpdateDirichletValues);

        mNewtonSolver->appendDirichletValues(tNewtonUpdateDirichletValues);
    }

    /***************************************************************************//**
     * \brief Update projected pressure gradient.
     * \param [in]     aControls  1-D view of controls, e.g. design variables
     * \param [in/out] aStateData data manager with current and previous global and local state data
    *******************************************************************************/
    void updateProjectedPressureGradient(const Plato::ScalarVector &aControls,
                                         Plato::CurrentStates &aStateData)
    {
        Plato::OrdinalType tNextStepIndex = aStateData.mCurrentStepIndex + static_cast<Plato::OrdinalType>(1);
        if(tNextStepIndex >= mTimeData->mNumTimeSteps)
        {
            return;
        }

        // copy projection state, i.e. pressure
        Plato::blas1::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, mPressure);

        // compute projected pressure gradient
        auto tNextProjectedPressureGradient = Kokkos::subview(mProjectedPressGrad, tNextStepIndex, Kokkos::ALL());
        Plato::blas1::fill(0.0, tNextProjectedPressureGradient);
        auto tProjResidual = mProjectionEquation->value(tNextProjectedPressureGradient, mPressure, aControls, tNextStepIndex);
        auto tProjJacobian = mProjectionEquation->gradient_u(tNextProjectedPressureGradient, mPressure, aControls, tNextStepIndex);
        Plato::blas1::scale(-1.0, tProjResidual);
        Plato::Solve::RowSummed<PhysicsT::mNumSpatialDims>(tProjJacobian, tNextProjectedPressureGradient, tProjResidual);
    }

    /***************************************************************************//**
     * \brief Get previous state
     * \param [in]     aCurrentStepIndex current time step index
     * \param [in]     aStates           states at each time step
     * \param [in/out] aOutput           previous state
    *******************************************************************************/
    void getPreviousState(const Plato::OrdinalType & aCurrentStepIndex,
                          const Plato::ScalarMultiVector & aStates,
                          Plato::ScalarVector & aOutput) const
    {
        auto tPreviousStepIndex = aCurrentStepIndex - static_cast<Plato::OrdinalType>(1);
        if(tPreviousStepIndex >= static_cast<Plato::OrdinalType>(0))
        {
            aOutput = Kokkos::subview(aStates, tPreviousStepIndex, Kokkos::ALL());
        }
        else
        {
            auto tLength = aStates.extent(1);
            aOutput = Plato::ScalarVector("State t=i-1", tLength);
            Plato::blas1::fill(0.0, aOutput);
        }
    }
    

    /***************************************************************************//**
     * \brief Evaluate criterion
     * \param [in] aCriterion   criterion scalar function interface
     * \param [in] aGlobalState global states for all time steps
     * \param [in] aLocalState  local states for all time steps
     * \param [in] aControls    current controls, e.g. design variables
     * \return new criterion value
    *******************************************************************************/
    Plato::Scalar evaluateCriterion(Plato::LocalScalarFunctionInc & aCriterion,
                                    const Plato::ScalarMultiVector & aGlobalState,
                                    const Plato::ScalarMultiVector & aLocalState,
                                    const Plato::ScalarVector & aControls)
    {
        Plato::ScalarVector tPreviousLocalState;
        Plato::ScalarVector tPreviousGlobalState;

        Plato::Scalar tOutput = 0;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mTimeData->mNumTimeSteps; tCurrentStepIndex++)
        {
            // SET CURRENT STATES
            auto tCurrentLocalState = Kokkos::subview(aLocalState, tCurrentStepIndex, Kokkos::ALL());
            auto tCurrentGlobalState = Kokkos::subview(aGlobalState, tCurrentStepIndex, Kokkos::ALL());

            // SET PREVIOUS AND FUTURE STATES
            this->getPreviousState(tCurrentStepIndex, aLocalState, tPreviousLocalState);
            this->getPreviousState(tCurrentStepIndex, aGlobalState, tPreviousGlobalState);
            this->setInitialTemperature(tCurrentStepIndex, tPreviousGlobalState);

            mTimeData->updateTimeData(tCurrentStepIndex);
            tOutput += aCriterion.value(tCurrentGlobalState, tPreviousGlobalState,
                                        tCurrentLocalState, tPreviousLocalState,
                                        aControls, *mTimeData);
        }

        return tOutput;
    }

    /***************************************************************************//**
     * \brief Add contribution from partial derivative of criterion with respect to
     * controls to total derivative of criterion with respect to controls.
     * \param [in]     aCriterion     design criterion interface
     * \param [in]     aControls      current controls, e.g. design variables
     * \param [in/out] aTotalGradient total derivative of criterion with respect to controls
    *******************************************************************************/
    void addCriterionPartialDerivativeZ(Plato::LocalScalarFunctionInc & aCriterion,
                                        const Plato::ScalarVector & aControls,
                                        Plato::ScalarVector & aTotalGradient)
    {
        Plato::ScalarVector tPreviousLocalState;
        Plato::ScalarVector tPreviousGlobalState;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mTimeData->mNumTimeSteps; tCurrentStepIndex++)
        {
            auto tCurrentLocalState = Kokkos::subview(mLocalStates, tCurrentStepIndex, Kokkos::ALL());
            auto tCurrentGlobalState = Kokkos::subview(mGlobalStates, tCurrentStepIndex, Kokkos::ALL());

            // SET PREVIOUS STATES
            this->getPreviousState(tCurrentStepIndex, mLocalStates, tPreviousLocalState);
            this->getPreviousState(tCurrentStepIndex, mGlobalStates, tPreviousGlobalState);
            this->setInitialTemperature(tCurrentStepIndex, tPreviousGlobalState);

            mTimeData->updateTimeData(tCurrentStepIndex);
            auto tDfDz = aCriterion.gradient_z(tCurrentGlobalState, tPreviousGlobalState,
                                               tCurrentLocalState, tPreviousLocalState,
                                               aControls, *mTimeData);
            mWorksetBase.assembleScalarGradientZ(tDfDz, aTotalGradient);
        }
    }

    /***************************************************************************//**
     * \brief Add contribution from partial derivative of criterion with respect to
     * configuration to total derivative of criterion with respect to configuration.
     * \param [in]     aCriterion     design criterion interface
     * \param [in]     aControls      current controls, e.g. design variables
     * \param [in/out] aTotalGradient total derivative of criterion with respect to configuration
    *******************************************************************************/
    void addCriterionPartialDerivativeX(Plato::LocalScalarFunctionInc & aCriterion,
                                        const Plato::ScalarVector & aControls,
                                        Plato::ScalarVector & aTotalGradient)
    {
        Plato::ScalarVector tPreviousLocalState;
        Plato::ScalarVector tPreviousGlobalState;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mTimeData->mNumTimeSteps; tCurrentStepIndex++)
        {
            auto tCurrentLocalState = Kokkos::subview(mLocalStates, tCurrentStepIndex, Kokkos::ALL());
            auto tCurrentGlobalState = Kokkos::subview(mGlobalStates, tCurrentStepIndex, Kokkos::ALL());

            // SET PREVIOUS AND FUTURE LOCAL STATES
            this->getPreviousState(tCurrentStepIndex, mLocalStates, tPreviousLocalState);
            this->getPreviousState(tCurrentStepIndex, mGlobalStates, tPreviousGlobalState);
            this->setInitialTemperature(tCurrentStepIndex, tPreviousGlobalState);

            mTimeData->updateTimeData(tCurrentStepIndex);
            auto tDfDX = aCriterion.gradient_x(tCurrentGlobalState, tPreviousGlobalState,
                                               tCurrentLocalState, tPreviousLocalState,
                                               aControls, *mTimeData);
            mWorksetBase.assembleVectorGradientX(tDfDX, aTotalGradient);
        }
    }

    /***************************************************************************//**
     * \brief Initialize adjoint solver, i.e. append necessary system of equation interfaces.
    *******************************************************************************/
    void initializeAdjointSolver()
    {
        mAdjointSolver->appendDirichletDofs(mDirichletDofs);
        mAdjointSolver->appendLocalEquation(mLocalEquation);
        mAdjointSolver->appendGlobalEquation(mGlobalEquation);
        mAdjointSolver->appendProjectionEquation(mProjectionEquation);
        mAdjointSolver->setNumPseudoTimeSteps(mTimeData->mNumTimeSteps);
    }

    /***************************************************************************//**
     * \brief Perform backward time integration and add Partial Differential Equation
     * (PDE) contribution to total gradient.
     * \param [in]     aType      partial derivative type
     * \param [in]     aControls current controls, e.g. design variables
     * \param [in/out] aOutput   total derivative of criterion with respect to controls
    *******************************************************************************/
    void backwardTimeIntegration(const Plato::PartialDerivative::derivative_t & aType,
                                 const Plato::ScalarVector & aControls,
                                 Plato::ScalarVector aTotalDerivative)
    {
        // Create state data manager
        auto tNumCells = mLocalEquation->numCells();
        Plato::ForwardStates tCurrentStates(aType, *mTimeData);
        Plato::ForwardStates tPreviousStates(aType, *mTimeData);
        Plato::AdjointStates tAdjointStates(mGlobalEquation->size(), mLocalEquation->size(), mProjectionEquation->size());
        tAdjointStates.mInvLocalJacT = ScalarArray3D("Inv(DhDc)^T", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        this->initializeAdjointSolver();

        // outer loop for pseudo time steps
        auto tLastStepIndex = mTimeData->mNumTimeSteps - static_cast<Plato::OrdinalType>(1);
        for(tCurrentStates.mCurrentStepIndex = tLastStepIndex; tCurrentStates.mCurrentStepIndex >= 0; tCurrentStates.mCurrentStepIndex--)
        {
            tCurrentStates.mTimeData.updateTimeData(tCurrentStates.mCurrentStepIndex);

            tPreviousStates.mCurrentStepIndex = tCurrentStates.mCurrentStepIndex + 1;
            tPreviousStates.mTimeData.updateTimeData(tPreviousStates.mCurrentStepIndex);
            if(tPreviousStates.mCurrentStepIndex < mTimeData->mNumTimeSteps)
            {
                this->updateForwardState(tPreviousStates);
            }
            this->updateForwardState(tCurrentStates);
            this->updateAdjointState(tAdjointStates);

            mAdjointSolver->updateAdjointVariables(aControls, tCurrentStates, tPreviousStates, tAdjointStates);
            mAdjointSolver->addContributionFromPDE(aControls, tCurrentStates, tAdjointStates, aTotalDerivative);
        }
    }

    /***************************************************************************//**
     * \brief Update state data for time step n, i.e. current time step:
     * \param [in] aStateData state data manager
     * \param [in] aZeroEntries flag - zero all entries in current states (default = false)
    *******************************************************************************/
    void cacheStateData(Plato::CurrentStates & aStateData)
    {
        // GET CURRENT STATE
        aStateData.mCurrentLocalState = Kokkos::subview(mLocalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mCurrentGlobalState = Kokkos::subview(mGlobalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mProjectedPressGrad = Kokkos::subview(mProjectedPressGrad, aStateData.mCurrentStepIndex, Kokkos::ALL());

        // GET PREVIOUS STATE
        this->getPreviousState(aStateData.mCurrentStepIndex, mLocalStates, aStateData.mPreviousLocalState);
        this->getPreviousState(aStateData.mCurrentStepIndex, mGlobalStates, aStateData.mPreviousGlobalState);

        // SET ENTRIES IN CURRENT STATES TO ZERO
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentLocalState);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentGlobalState);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mPressure);
    }

    /***************************************************************************//**
     * \brief Update state data for time step n, i.e. current time step:
     * \param [in] aStateData state data manager
     * \param [in] aZeroEntries flag - zero all entries in current states (default = false)
    *******************************************************************************/
    void updateForwardState(Plato::ForwardStates & aStateData)
    {
        // GET CURRENT STATE
        aStateData.mCurrentLocalState = Kokkos::subview(mLocalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mCurrentGlobalState = Kokkos::subview(mGlobalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mProjectedPressGrad = Kokkos::subview(mProjectedPressGrad, aStateData.mCurrentStepIndex, Kokkos::ALL());
        if(aStateData.mPressure.size() <= static_cast<Plato::OrdinalType>(0))
        {
            auto tNumVerts = mSpatialModel.Mesh->NumNodes();
            aStateData.mPressure = Plato::ScalarVector("Current Pressure Field", tNumVerts);
        }
        Plato::blas1::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, aStateData.mPressure);

        // GET PREVIOUS STATE.
        this->getPreviousState(aStateData.mCurrentStepIndex, mLocalStates, aStateData.mPreviousLocalState);
        this->getPreviousState(aStateData.mCurrentStepIndex, mGlobalStates, aStateData.mPreviousGlobalState);
        this->setInitialTemperature(aStateData.mCurrentStepIndex, aStateData.mPreviousGlobalState);
    }

    /***************************************************************************//**
     * \brief Update adjoint data for time step n, i.e. current time step:
     * \param [in] aAdjointData adjoint data manager
    *******************************************************************************/
    void updateAdjointState(Plato::AdjointStates& aAdjointStates)
    {
        // NOTE: CURRENT ADJOINT VARIABLES ARE UPDATED AT SOLVE TIME. THERE IS NO NEED TO SET THEM TO ZERO HERE.
        const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 0.0;
        Plato::blas1::update(tAlpha, aAdjointStates.mCurrentLocalAdjoint, tBeta, aAdjointStates.mPreviousLocalAdjoint);
        Plato::blas1::update(tAlpha, aAdjointStates.mCurrentGlobalAdjoint, tBeta, aAdjointStates.mPreviousGlobalAdjoint);
        Plato::blas1::update(tAlpha, aAdjointStates.mProjPressGradAdjoint, tBeta, aAdjointStates.mPreviousProjPressGradAdjoint);
    }

    /***************************************************************************//**
     * \brief Allocate objective function interface and adjoint containers
     * \param [in] aInputParams input parameters database
    *******************************************************************************/
    void allocateCriteria(Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isSublist("Criteria"))
        {
            Plato::PathDependentScalarFunctionFactory<PhysicsT> tObjectiveFunctionFactory;

            auto tCriteriaParams = aInputParams.sublist("Criteria");
            for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
                std::string tName = tCriteriaParams.name(tIndex);

                TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
                  " Parameter in Criteria block not valid.  Expect lists only.");

                if( tCriteriaParams.sublist(tName).get<bool>("Linear", false) == false )
                {
                    auto tCriterion = tObjectiveFunctionFactory.create(mSpatialModel, mDataMap, aInputParams, tName);
                    if( tCriterion != nullptr )
                    {
                        mCriteria[tName] = tCriterion;
                    }
                }
            }
        }
        if(mCriteria.size() == 0)
        {
            REPORT("Plasticity Problem: No criteria defined for this problem.")
        }
    }
    /******************************************************************************/ /**
    * \brief Return solution database.
    * \return solution database
    **********************************************************************************/
    Plato::Solutions getSolution() const override
    {
        Plato::Solutions tSolution(mPhysics, mPDEType);
        tSolution.set("State", mGlobalStates);
        tSolution.setDofNames("State", mGlobalEquation->getDofNames());
        return tSolution;
    }
};
// class PlasticityProblem

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<1>>;
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<2>>;
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<3>>;
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif

