#ifndef PLATO_PROBLEM_PARABOLIC_PROBLEM_DECL
#define PLATO_PROBLEM_PARABOLIC_PROBLEM_DECL

#include <fstream>
#include <map>
#include <memory>
#include <optional>

#include "boundary_conditions/EssentialBCs.hpp"
#include "core_types/PlatoTypes.hpp"
#include "domain/Solutions.hpp"
#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "problem/PlatoAbstractProblem.hpp"
#include "problem/parabolic/ScalarFunctionBase.hpp"
#include "problem/parabolic/TrapezoidIntegrator.hpp"
#include "problem/parabolic/VectorFunction.hpp"
#include "solver/PlatoAbstractSolver.hpp"
#include "solver/nonlinear_solvers/NewtonSolver.hpp"
#include "utilities/ParallelComm.hpp"

namespace plato::parabolic
{
/// @brief class to manage solution of PDE, computation of criterion values, and computation of criterion gradients.
/// @tparam PhysicsType struct specifying element and function factory types for physics.
template <typename PhysicsType>
class Problem : public Plato::AbstractProblem
{
   private:
    using VectorFunctionType = Plato::Parabolic::VectorFunction<PhysicsType>;
    using ElementType = typename PhysicsType::ElementType;

    using Criterion = std::shared_ptr<Plato::Parabolic::ScalarFunctionBase>;

   public:
    Problem(Plato::Mesh aMesh, Teuchos::ParameterList& aProblemParams, Plato::Comm::Machine aMachine);

    /// @brief apply essential boundary conditions as state constraints on the linear system matrix @a aMatrix and
    /// vector @a aVector.
    /// Essential boundary condition values will be scaled by @a aScale.
    void applyStateConstraints(const Teuchos::RCP<Plato::CrsMatrixType>& aMatrix,
                               const Plato::ScalarVector& aVector,
                               Plato::Scalar aScale);

    /// @brief write solution fields to output file with path @a FilePath.
    void output(const std::string& aFilepath) override final;

    /// @brief update criteria with control values @a aControl and state stored in @a aSolution.
    /// @note this implementation is currently a no-op
    void updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution) override final;

    /// @brief solve the PDE forward problem using control values @a aControl.
    Plato::Solutions solution(const Plato::ScalarVector& aControl) override final;

    /// @brief compute the value of criterion with name @a aName using control values @a aControl and state stored in @a
    /// aSolution.
    Plato::Scalar criterionValue(const Plato::ScalarVector& aControl,
                                 const Plato::Solutions& aSolution,
                                 const std::string& aName) override final;

    /// @brief compute the gradient w.r.t control of criterion with name @a aName using control values @a aControl and
    /// state stored in @a aSolution.
    Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                          const Plato::Solutions& aSolution,
                                          const std::string& aName) override final;

    /// @brief implementation of gradient computation w.r.t control for criterion @a aCriterion using control values @a
    /// aControl and state stored in @a aSolution.
    Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                          const Plato::Solutions& aSolution,
                                          Criterion aCriterion);

    /// @brief compute the gradient w.r.t nodal coordinates of criterion with name @a aName using control values
    /// @a aControl and state stored in @a aSolution.
    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                           const Plato::Solutions& aSolution,
                                           const std::string& aName) override final;

    /// @brief implementation of gradient computation w.r.t nodal coordinates for criterion @a aCriterion using control
    /// values @a aControl and state stored in @a aSolution.
    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                           const Plato::Solutions& aSolution,
                                           Criterion aCriterion);

    /// @brief returns the state stored in mState and mStateDot.
    Plato::Solutions getSolution() const override;

   private:
    plato::domain::SpatialModel mSpatialModel;
    std::shared_ptr<VectorFunctionType> mPDE;
    std::string mPDEType;
    std::string mPhysics;
    std::optional<std::ofstream> mOutputFileStream;
    std::ostream& mOutputStream;
    Plato::Parabolic::TrapezoidIntegrator mTrapezoidIntegrator;
    Plato::OrdinalType mNumSteps;
    Plato::Scalar mTimeStep;
    Plato::ScalarMultiVector mState;
    Plato::ScalarMultiVector mStateDot;
    bool mSaveState;
    Plato::EssentialBCs<ElementType> mEssentialBCs;
    std::shared_ptr<Plato::MultipointConstraints> mMPCs;
    std::map<std::string, Criterion> mCriteriaMap;
    Plato::rcp<Plato::AbstractSolver> mSolver;
    algorithms::nonlinear_solvers::NewtonSolver mNewtonSolver;
    Plato::ScalarMultiVector mAdjointStates;
    Plato::ScalarMultiVector mAdjointStatesV;
};

}  // namespace plato::parabolic

#endif
