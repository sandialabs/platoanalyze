#pragma once

#include <memory>
#include <sstream>

#include "boundary_conditions/ApplyConstraints.hpp"
#include "domain/AnalyzeOutput.hpp"
#include "domain/Solutions.hpp"
#include "domain/SpatialModel.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoMathHelpers.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/ImplicitFunctors.hpp"
#include "parsing/ParseTools.hpp"
#include "problem/PlatoAbstractProblem.hpp"
#include "problem/helmholtz/VectorFunction.hpp"
#include "solver/PlatoSolverFactory.hpp"
#include "utilities/AnalyzeMacros.hpp"
#include "utilities/ParallelComm.hpp"
#include "utilities/PlatoUtilities.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
/**
 * \brief Manage scalar and vector function evaluations
 **********************************************************************************/
template <typename PhysicsType>
class Problem : public Plato::AbstractProblem
{
   private:
    using ElementType = typename PhysicsType::ElementType;

    using VectorFunctionType = Plato::Helmholtz::VectorFunction<PhysicsType>;

    plato::domain::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    rcp<Plato::AbstractSolver> mSolver;

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

    auto pde() -> VectorFunctionType&;
    auto solver() -> Plato::AbstractSolver&;

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
     * \return solution database
     **********************************************************************************/
    Plato::Solutions solution(const Plato::ScalarVector& aControl) override final;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
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
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
     **********************************************************************************/
    Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                          const Plato::Solutions& aSolution,
                                          const std::string& aName) override final;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
     **********************************************************************************/
    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                           const Plato::Solutions& aSolution,
                                           const std::string& aName) override final;

    /******************************************************************************/
    /**
     * \brief Return solution database.
     * \return solution database
     **********************************************************************************/
    Plato::Solutions getSolution() const override final;
};
// class Problem

}  // namespace Helmholtz

}  // namespace Plato
