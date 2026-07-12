#ifndef PLATO_GEOMETRIC_PROBLEM_DECL_H
#define PLATO_GEOMETRIC_PROBLEM_DECL_H

#include <Teuchos_ParameterList.hpp>
#include <filesystem>
#include <map>
#include <memory>
#include <string>

#include "domain/Solutions.hpp"
#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "problem/PlatoAbstractProblem.hpp"
#include "problem/geometric/ScalarFunctionBase.hpp"
#include "utilities/ParallelComm.hpp"

namespace plato::problem::geometric
{
/// @brief class to manage computation of criterion values and gradients for geometric quantities.
/// Since geometric quantities require no state computation, this class solves no PDE.
template <typename PhysicsType>
class Problem : public Plato::AbstractProblem
{
   private:
    using ElementType = typename PhysicsType::ElementType;
    using TopoElementType = typename ElementType::TopoElementType;

    using Criterion = std::shared_ptr<Plato::Geometric::ScalarFunctionBase>;

   public:
    Problem(Plato::Mesh aMesh, Teuchos::ParameterList& aProblemParams, Plato::Comm::Machine aMachine);

    /// @brief update criteria with control values @a aControl. State stored in @a aSolution is not used in this
    /// implementation.
    void updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution) override final;

    /// @brief virtual function for solving the PDE forward problem using control values @a aControl.
    /// This function implementation is a no-op since no state is needed to compute geometric quantities.
    Plato::Solutions solution(const Plato::ScalarVector& aControl) override final;

    /// @brief compute the value of criterion with name @a aName using control values @a aControl. State stored in @a
    /// aSolution is not used in this implementation.
    Plato::Scalar criterionValue(const Plato::ScalarVector& aControl,
                                 const Plato::Solutions& aSolution,
                                 const std::string& aName) override final;

    /// @brief compute the gradient w.r.t control of criterion with name @a aName using control values @a aControl.
    /// State stored in @a aSolution is not used in this implementation.
    Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                          const Plato::Solutions& aSolution,
                                          const std::string& aName) override final;

    /// @brief compute the gradient w.r.t nodal coordinates of criterion with name @a aName using control values
    /// @a aControl. State stored in @a aSolution is not used in this implementation.
    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                           const Plato::Solutions& aSolution,
                                           const std::string& aName) override final;

    /// @brief returns the state stored in mState.
    /// This function implementation is a no-op since no state is needed to compute geometric quantities.
    Plato::Solutions getSolution() const override final;

    /// @brief write solution fields to output file with path @a FilePath.
    /// This function implementation is a no-op since no state is needed to compute geometric quantities.
    void output(const std::filesystem::path& aFilepath) const override final;

   private:
    plato::domain::SpatialModel mSpatialModel;
    std::map<std::string, Criterion> mCriteriaMap;
};
}  // namespace plato::problem::geometric

#endif
