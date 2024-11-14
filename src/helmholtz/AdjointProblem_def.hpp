#pragma once

#include "helmholtz/AdjointProblem_decl.hpp"

namespace Plato::Helmholtz {

template <typename PhysicsType>
AdjointProblem<PhysicsType>::AdjointProblem(Plato::Mesh aMesh,
                                            Teuchos::ParameterList& aProblemParams,
                                            Comm::Machine aMachine)
    : mHelmholtzProblem{std::make_shared<Problem<PhysicsType>>(std::move(aMesh), aProblemParams, std::move(aMachine))} {
}

template <typename PhysicsType>
AdjointProblem<PhysicsType>::AdjointProblem(std::shared_ptr<Plato::Helmholtz::Problem<PhysicsType>> aProblem)
    : mHelmholtzProblem{std::move(aProblem)} {
  assert(mHelmholtzProblem);
}

template <typename PhysicsType>
Plato::OrdinalType AdjointProblem<PhysicsType>::numNodes() const {
  return mHelmholtzProblem->numNodes();
}

template <typename PhysicsType>
Plato::OrdinalType AdjointProblem<PhysicsType>::numCells() const {
  return mHelmholtzProblem->numCells();
}

template <typename PhysicsType>
Plato::OrdinalType AdjointProblem<PhysicsType>::numDofsPerCell() const {
  return mHelmholtzProblem->numDofsPerCell();
}

template <typename PhysicsType>
Plato::OrdinalType AdjointProblem<PhysicsType>::numNodesPerCell() const {
  return mHelmholtzProblem->numNodesPerCell();
}

template <typename PhysicsType>
Plato::OrdinalType AdjointProblem<PhysicsType>::numDofsPerNode() const {
  return mHelmholtzProblem->numDofsPerNode();
}

template <typename PhysicsType>
Plato::OrdinalType AdjointProblem<PhysicsType>::numControlsPerNode() const {
  return mHelmholtzProblem->numControlsPerNode();
}

template <typename PhysicsType>
void AdjointProblem<PhysicsType>::output(const std::string& aFilepath) {
  mHelmholtzProblem->output(aFilepath);
}

template <typename PhysicsType>
void AdjointProblem<PhysicsType>::updateProblem(const Plato::ScalarVector& aControl,
                                                const Plato::Solutions& aSolution) {
  mHelmholtzProblem->updateProblem(aControl, aSolution);
}

template <typename PhysicsType>
Plato::Solutions AdjointProblem<PhysicsType>::solution(const Plato::ScalarVector& aControl)
{
    return mHelmholtzProblem->solution(aControl);
}

template <typename PhysicsType>
Plato::ScalarVector AdjointProblem<PhysicsType>::criterionGradient(const Plato::ScalarVector& aControl,
                                                                   const std::string& aName) {
    // Given `K \rho - M z = 0`, \rho = K^-1 M z
    // The Jacobian J of \rho with respect to z is then K^-1 M
    // This computes the product of J v, which corresponds to v^T J^T.
    const auto& tPDE = mHelmholtzProblem->pde();
    Plato::ScalarVector tSolution("derivative of criterion wrt unfiltered control", tPDE.size());
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tSolution);

    const auto tJacobian = tPDE.gradient_u(tSolution, aControl);

    auto tPartialPDE_WRT_Control = tPDE.gradient_z(tSolution, aControl);

    Plato::blas1::scale(-1.0, aControl);

    Plato::ScalarVector tIntermediateSolution("intermediate solution", tPDE.size());
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tIntermediateSolution);

    Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, aControl, tIntermediateSolution);
    mHelmholtzProblem->solver().solve(*tJacobian, tSolution, tIntermediateSolution);

    return tSolution;
}

template <typename PhysicsType>
Plato::Scalar AdjointProblem<PhysicsType>::criterionValue(const Plato::ScalarVector& aControl,
                                                          const std::string& aName) {
    return mHelmholtzProblem->criterionValue(aControl, aName);
}

template <typename PhysicsType>
Plato::Scalar AdjointProblem<PhysicsType>::criterionValue(const Plato::ScalarVector& aControl,
                                                          const Plato::Solutions& aSolution,
                                                          const std::string& aName) {
    return mHelmholtzProblem->criterionValue(aControl, aSolution, aName);
}

template <typename PhysicsType>
Plato::ScalarVector AdjointProblem<PhysicsType>::criterionGradient(const Plato::ScalarVector& aControl,
                                                                   const Plato::Solutions& aSolution,
                                                                   const std::string& aName) {
    return mHelmholtzProblem->criterionGradient(aControl, aSolution, aName);
}

template <typename PhysicsType>
Plato::ScalarVector AdjointProblem<PhysicsType>::criterionGradientX(const Plato::ScalarVector& aControl,
                                                                    const Plato::Solutions& aSolution,
                                                                    const std::string& aName) {
    return mHelmholtzProblem->criterionGradientX(aControl, aSolution, aName);
}

template <typename PhysicsType>
Plato::ScalarVector AdjointProblem<PhysicsType>::criterionGradientX(const Plato::ScalarVector& aControl,
                                                                    const std::string& aName) {
    return mHelmholtzProblem->criterionGradientX(aControl, aName);
}

template <typename PhysicsType>
Plato::Solutions AdjointProblem<PhysicsType>::getSolution() const
{
    return mHelmholtzProblem->getSolution();
}

}  // namespace Plato::Helmholtz
