#pragma once

#include <Teuchos_RCPDecl.hpp>

#include "boundary_conditions/EssentialBCs.hpp"
#include "boundary_conditions/NaturalBCs.hpp"
#include "domain/Solutions.hpp"
#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoMathHelpers.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "parsing/ParseTools.hpp"
#include "problem/Geometrical.hpp"
#include "problem/PlatoAbstractProblem.hpp"
#include "problem/PlatoSequence.hpp"
#include "problem/elliptic/hatching/ScalarFunctionBase.hpp"
#include "problem/elliptic/hatching/StateUpdate.hpp"
#include "problem/elliptic/hatching/VectorFunction.hpp"
#include "problem/geometric/ScalarFunctionBase.hpp"
#include "solver/PlatoAbstractSolver.hpp"
#include "utilities/ParallelComm.hpp"

namespace Plato
{

namespace Elliptic
{

namespace Hatching
{

/******************************************************************************/
/**
 * \brief Manage scalar and vector function evaluations
 **********************************************************************************/
template <typename PhysicsType>
class Problem : public Plato::AbstractProblem
{
   private:
    using Criterion = std::shared_ptr<Plato::Elliptic::Hatching::ScalarFunctionBase>;
    using Criteria = std::map<std::string, Criterion>;

    using LinearCriterion = std::shared_ptr<Plato::Geometric::ScalarFunctionBase>;
    using LinearCriteria = std::map<std::string, LinearCriterion>;

    using ElementType = typename PhysicsType::ElementType;
    using TopoElementType = typename ElementType::TopoElementType;

    using VectorFunctionType = Plato::Elliptic::Hatching::VectorFunction<PhysicsType>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    Plato::Sequence<ElementType> mSequence;

    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    LinearCriteria mLinearCriteria;
    Criteria mCriteria;

    Plato::OrdinalType mNumNewtonSteps;
    Plato::Scalar mNewtonResTol, mNewtonIncTol;

    bool mSaveState;

    Plato::ScalarMultiVector mGlobalAdjoints;
    Plato::ScalarMultiVector mLocalAdjoints;
    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mGlobalStates;
    Plato::ScalarMultiVector mTotalStates;
    Plato::ScalarArray4D mLocalStates;

    bool mIsSelfAdjoint; /*!< indicates if problem is self-adjoint */

    Teuchos::RCP<Plato::CrsMatrixType> mGlobalJacobian; /*!< Global jacobian matrix */
    Teuchos::RCP<Plato::CrsMatrixType> mLocalJacobian;  /*!< Global jacobian matrix */

    Plato::OrdinalVector mBcDofs;  /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

    rcp<Plato::AbstractSolver> mSolver;

    Plato::StateUpdate<PhysicsType> mStateUpdate;

    std::string mPDEType; /*!< partial differential equation type */
    std::string mPhysics; /*!< physics used for the simulation */

   public:
    /******************************************************************************/
    /**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aProblemParams input parameters database
     **********************************************************************************/
    Problem(Plato::Mesh aMesh, Teuchos::ParameterList& aProblemParams, Comm::Machine aMachine);

    Plato::OrdinalType numNodes() const;

    Plato::OrdinalType numCells() const;

    Plato::OrdinalType numDofsPerCell() const;

    Plato::OrdinalType numNodesPerCell() const;

    Plato::OrdinalType numDofsPerNode() const;

    Plato::OrdinalType numControlsPerNode() const;

    /******************************************************************************/
    /**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
     **********************************************************************************/
    void applyStateConstraints(const Teuchos::RCP<Plato::CrsMatrixType>& aMatrix,
                               const Plato::ScalarVector& aVector,
                               Plato::Scalar aScale);

    /******************************************************************************/
    /**
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
     **********************************************************************************/
    void output(const std::string& aFilepath) override final;

    /******************************************************************************/
    /**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution) override final;

    /******************************************************************************/
    /**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return Plato::Solution composed of state variables
     ***********************************************************************************/
    Plato::Solutions solution(const Plato::ScalarVector& aControl) override final;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
     **********************************************************************************/
    Plato::Scalar criterionValue(const Plato::ScalarVector& aControl,
                                 const Plato::Solutions& aSolution,
                                 const std::string& aName) override final;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution containing state
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
     **********************************************************************************/
    Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                          const Plato::Solutions& aSolution,
                                          const std::string& aName) override final;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aCriterion criterion to be evaluated
     * \return 1D view - criterion gradient wrt control variables
     **********************************************************************************/
    Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                          const Plato::Solutions& aSolution,
                                          Criterion aCriterion);

    /******************************************************************************/
    /**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution containing state
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
     **********************************************************************************/
    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                           const Plato::Solutions& aSolution,
                                           const std::string& aName) override final;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \param [in] aCriterion criterion to be evaluated
     * \return 1D view - criterion gradient wrt configuration variables
     **********************************************************************************/
    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                           const Plato::Solutions& aSolution,
                                           Criterion aCriterion);

    /***************************************************************************/
    /**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aProblemParams input parameters database
     *******************************************************************************/
    void readEssentialBoundaryConditions(Teuchos::ParameterList& aProblemParams);

    /***************************************************************************/
    /**
     * \brief Set essential (Dirichlet) boundary conditions
     * \param [in] aDofs   degrees of freedom associated with Dirichlet boundary conditions
     * \param [in] aValues values associated with Dirichlet degrees of freedom
     *******************************************************************************/
    void setEssentialBoundaryConditions(const Plato::OrdinalVector& aDofs, const Plato::ScalarVector& aValues);

   private:
    /******************************************************************************/
    /**
     * \brief Initialize member data
     * \param [in] aProblemParams input parameters database
     **********************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams);

    void applyAdjointConstraints(const Teuchos::RCP<Plato::CrsMatrixType>& aMatrix, const Plato::ScalarVector& aVector);

    /******************************************************************************/ /**
                                                                                      * \brief Return solution database.
                                                                                      * \return solution database
                                                                                      **********************************************************************************/
    Plato::Solutions getSolution() const override final;
};
// class Problem

}  // namespace Hatching

}  // namespace Elliptic

}  // namespace Plato
