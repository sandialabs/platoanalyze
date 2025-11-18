#pragma once

#include "problem/helmholtz/Problem_decl.hpp"
#include "problem/helmholtz/VectorFunction.hpp"

namespace Plato::Helmholtz
{
/******************************************************************************/
/**
 * \brief Manage scalar and vector function evaluations and implement the gradient
 *   as multiplication with the adjoint Jacobian.
 **********************************************************************************/
template <typename PhysicsType>
class AdjointProblem : public Plato::AbstractProblem
{
   private:
    using ElementType = typename PhysicsType::ElementType;
    using VectorFunctionType = Plato::Helmholtz::VectorFunction<PhysicsType>;

   public:
    AdjointProblem(Plato::Mesh aMesh, Teuchos::ParameterList& aProblemParams, Comm::Machine aMachine);
    AdjointProblem(std::shared_ptr<Plato::Helmholtz::Problem<PhysicsType>> aProblem);

    Plato::OrdinalType numNodes() const;
    Plato::OrdinalType numCells() const;
    Plato::OrdinalType numDofsPerCell() const;
    Plato::OrdinalType numNodesPerCell() const;
    Plato::OrdinalType numDofsPerNode() const;
    Plato::OrdinalType numControlsPerNode() const;

    void output(const std::string& aFilepath) override final;
    void updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution) override final;

    Plato::Solutions solution(const Plato::ScalarVector& aControl) override final;

    Plato::Scalar criterionValue(const Plato::ScalarVector& aControl,
                                 const Plato::Solutions& aSolution,
                                 const std::string& aName) override final;

    Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                          const Plato::Solutions& aSolution,
                                          const std::string& aName) override final;

    Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                           const Plato::Solutions& aSolution,
                                           const std::string& aName) override final;

    Plato::Solutions getSolution() const override final;

   private:
    std::shared_ptr<Plato::Helmholtz::Problem<PhysicsType>> mHelmholtzProblem;
};

}  // namespace Plato::Helmholtz
