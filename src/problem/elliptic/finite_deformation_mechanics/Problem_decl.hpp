#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_PROBLEM_DECL_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_PROBLEM_DECL_H

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <fstream>
#include <memory>
#include <optional>
#include <string>

#include "boundary_conditions/EssentialBCs.hpp"
#include "core_types/PlatoTypes.hpp"
#include "domain/Solutions.hpp"
#include "domain/SpatialModel.hpp"
#include "linear_algebra/CrsMatrix.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "problem/PlatoAbstractProblem.hpp"
#include "problem/elliptic/ScalarFunctionBase.hpp"
#include "problem/elliptic/VectorFunction.hpp"
#include "solver/PlatoAbstractSolver.hpp"
#include "utilities/ParallelComm.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
/// @brief class to manage solution of PDE, computation of criterion values, and computation of criterion gradients.
/// @tparam PhysicsType struct specifying element and function factory types for physics.
template <typename PhysicsType>
class Problem : public Plato::AbstractProblem
{
   private:
    using VectorFunctionType = Plato::Elliptic::VectorFunction<PhysicsType>;
    using ElementType = typename PhysicsType::ElementType;

    using Criterion = std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>;

   public:
    Problem(Plato::Mesh aMesh, Teuchos::ParameterList& aProblemParams, Plato::Comm::Machine aMachine);

    /// @brief update criteria with control values @a aControl and state stored in @a aSolution.
    void updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution) override final;

    /// @brief solve the PDE forward problem using control values @a aControl.
    Plato::Solutions solution(const Plato::ScalarVector& aControl) override final;

    /// @brief compute the value of criterion with name @a aName using control values @a aControl and state stored in @a
    /// aSolution.
    /// This is the preferred overload.
    Plato::Scalar criterionValue(const Plato::ScalarVector& aControl,
                                 const Plato::Solutions& aSolution,
                                 const std::string& aName) override final;

    /// @brief compute the gradient w.r.t control of criterion with name @a aName using control values @a aControl and
    /// state stored in @a aSolution.
    /// This is the preferred overload.
    Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                          const Plato::Solutions& aSolution,
                                          const std::string& aName) override final;

    /// @brief compute the gradient w.r.t nodal coordinates of criterion with name @a aName using control values
    /// @a aControl and state stored in @a aSolution.
    /// This is the preferred overload.
    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                           const Plato::Solutions& aSolution,
                                           const std::string& aName) override final;

    /// @brief returns the state stored in mState.
    Plato::Solutions getSolution() const override final;

    /// @brief write solution fields to output file with path @a FilePath.
    void output(const std::string& aFilepath) override final;

   private:
    /// @brief extract the last time step state from @a aSolution and output it as a new Plato::Solutions object.
    Plato::Solutions extractLastTimeStepSolution(const Plato::Solutions& aSolution);

    /// @brief generic function to compute gradients of criterion @a aCriterion that can be used for both control and
    /// config gradients.
    /// @tparam CriterionDerivativeFunc callable for computing the partial derivative of criterion w.r.t. the argument.
    ///         Must have the following signature:
    ///         Plato::ScalarVector(const Plato::Solutions& aSolutions, const Plato::ScalarVector& aControl)
    /// @tparam ResidualDerivativeFunc callable for computing the partial derivative of residual w.r.t. the argument.
    ///         Must have the following signature:
    ///         Teuchos::RCP<Plato::CrsMatrixType>(const Plato::ScalarVector& aState, const Plato::ScalarVector&
    ///         aControl)
    template <typename CriterionDerivativeFunc, typename ResidualDerivativeFunc>
    Plato::ScalarVector computeGradient(const Criterion& aCriterion,
                                        const Plato::ScalarVector& aControl,
                                        const Plato::Solutions& aSolution,
                                        const CriterionDerivativeFunc& aComputeCriterionDerivative,
                                        const ResidualDerivativeFunc& aComputeResidualDerivative);

   private:
    plato::domain::SpatialModel mSpatialModel;
    std::shared_ptr<VectorFunctionType> mPDE;
    std::string mPDEType;
    std::string mPhysics;
    std::optional<std::ofstream> mOutputFileStream;
    std::ostream& mOutputStream;
    Plato::OrdinalType mNumSteps;
    Plato::Scalar mTimeStep;
    Plato::OrdinalType mNumNewtonSteps;
    Plato::Scalar mNewtonIncTol;
    Plato::Scalar mNewtonResTol;
    Plato::ScalarMultiVector mState;
    bool mSaveState;
    bool mIsSelfAdjoint;
    Plato::EssentialBCs<ElementType> mEssentialBCs;
    Plato::rcp<Plato::AbstractSolver> mSolver;
    std::map<std::string, Criterion> mCriteriaMap;
};
}  // namespace plato::elliptic::finite_deformation_mechanics
#endif
