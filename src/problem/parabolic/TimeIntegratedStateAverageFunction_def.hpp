#ifndef PLATO_PARABOLIC_TIMEINTEGRATEDSTATEAVERAGEFUNCTION_DEF_H
#define PLATO_PARABOLIC_TIMEINTEGRATEDSTATEAVERAGEFUNCTION_DEF_H

#include <Teuchos_ParameterList.hpp>
#include <string>

#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/parabolic/TimeIntegratedStateAverageFunction_decl.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace plato::parabolic
{
template <typename PhysicsType>
TimeIntegratedStateAverageFunction<PhysicsType>::TimeIntegratedStateAverageFunction(
    const plato::domain::SpatialModel& aSpatialModel,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aProblemParams,
    const std::string& aName)
    : Plato::WorksetBase<ElementType>(aSpatialModel.mMesh), mName(aName), mSpatialModel(aSpatialModel)
{
    const auto tCriterionParams = aProblemParams.sublist("Criteria").sublist(mName);

    const auto tNodeSet = tCriterionParams.get<std::string>("Nodeset");
    const auto tNodeSetsInMesh = aSpatialModel.mMesh->GetNodeSetNames();
    const auto tNodeSetIterator = std::find(tNodeSetsInMesh.begin(), tNodeSetsInMesh.end(), tNodeSet);
    if (tNodeSetIterator == tNodeSetsInMesh.end())
    {
        ANALYZE_THROWERR(std::string("Nodeset with name '") + tNodeSet +
                         std::string("' specified in the 'Criteria' with name '") + aName +
                         std::string("' does not exist in mesh."))
    }
    mNodeSet = tNodeSet;

    const auto tStateComponent = tCriterionParams.get<int>("State Component");
    if (tStateComponent < 0 || tStateComponent > mNumDofsPerNode - 1)
    {
        ANALYZE_THROWERR(
            std::string("'State Component' specified in the 'Criteria' with name '") + aName +
            std::string("' is out of the range of number of degrees of freedom per node for the specified physics."))
    }
    mStateComponent = tStateComponent;
}

template <typename PhysicsType>
Plato::Scalar TimeIntegratedStateAverageFunction<PhysicsType>::value(const Plato::Solutions& aSolution,
                                                                     const Plato::ScalarVector& aControl,
                                                                     const Plato::Scalar aTimeStep) const
{
    const auto tNodeIds = mSpatialModel.mMesh->GetNodeSetNodes(mNodeSet);
    const auto tNumNodes = tNodeIds.size();

    const auto tStates = aSolution.get("State");
    const auto tNumSteps = tStates.extent(0);

    Plato::Scalar tReturnValue(0.0);
    const auto tNumDofsPerNode = mNumDofsPerNode;
    const auto tStateDof = mStateComponent;
    for (Plato::OrdinalType tStepIndex = 1; tStepIndex < tNumSteps; ++tStepIndex)
    {
        Plato::Scalar tNodalSum(0.0);
        const auto tStepState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<>(0, tNumNodes),
            KOKKOS_LAMBDA(const Plato::OrdinalType aNodeOrdinal, Plato::Scalar& aSum) {
                const auto tIndex = tNodeIds[aNodeOrdinal];
                aSum += tStepState(tNumDofsPerNode * tIndex + tStateDof);
            },
            tNodalSum);

        const auto tTrapezoidIntegrationConstant =
            tStepIndex == 1 || tStepIndex == tNumSteps - 1 ? 0.5 * aTimeStep : aTimeStep;
        tReturnValue += tTrapezoidIntegrationConstant * tNodalSum;
    }

    return tReturnValue / tNumNodes;
}

template <typename PhysicsType>
Plato::ScalarVector TimeIntegratedStateAverageFunction<PhysicsType>::gradient_u(const Plato::Solutions& aSolution,
                                                                                const Plato::ScalarVector& aControl,
                                                                                const Plato::OrdinalType aStepIndex,
                                                                                const Plato::Scalar aTimeStep) const
{
    const auto tNodeIds = mSpatialModel.mMesh->GetNodeSetNodes(mNodeSet);
    const auto tNumNodes = tNodeIds.size();

    const auto tStates = aSolution.get("State");
    assert(tStates.extent(0) > 0);
    assert(aStepIndex < tStates.extent(0));

    const Plato::ScalarVector tGradientU("gradient w.r.t. state", mNumDofsPerNode * mNumNodes);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tGradientU);

    const auto tNumDofsPerNode = mNumDofsPerNode;
    const auto tStateDof = mStateComponent;
    Kokkos::parallel_for(
        "gradient w.r.t. state", Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes),
        KOKKOS_LAMBDA(const Plato::OrdinalType aNodeOrdinal) {
            const auto tIndex = tNodeIds[aNodeOrdinal];
            tGradientU(tNumDofsPerNode * tIndex + tStateDof) = 1.0 / tNumNodes;
        });

    auto tNumSteps = tStates.extent(0);
    const auto tTrapezoidIntegrationConstant =
        aStepIndex == 1 || aStepIndex == tNumSteps - 1 ? 0.5 * aTimeStep : aTimeStep;
    Plato::blas1::scale(tTrapezoidIntegrationConstant, tGradientU);
    return tGradientU;
}

template <typename PhysicsType>
Plato::ScalarVector TimeIntegratedStateAverageFunction<PhysicsType>::gradient_v(const Plato::Solutions& aSolution,
                                                                                const Plato::ScalarVector& aControl,
                                                                                const Plato::OrdinalType aStepIndex,
                                                                                const Plato::Scalar aTimeStep) const
{
    const Plato::ScalarVector tGradientV("gradient w.r.t state dot", mNumDofsPerNode * mNumNodes);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tGradientV);
    return tGradientV;
}

template <typename PhysicsType>
Plato::ScalarVector TimeIntegratedStateAverageFunction<PhysicsType>::gradient_z(const Plato::Solutions& aSolution,
                                                                                const Plato::ScalarVector& aControl,
                                                                                const Plato::Scalar aTimeStep) const
{
    const Plato::ScalarVector tGradientZ("gradient w.r.t control", mNumNodes);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tGradientZ);
    return tGradientZ;
}

template <typename PhysicsType>
Plato::ScalarVector TimeIntegratedStateAverageFunction<PhysicsType>::gradient_x(const Plato::Solutions& aSolution,
                                                                                const Plato::ScalarVector& aControl,
                                                                                const Plato::Scalar aTimeStep) const
{
    const Plato::ScalarVector tGradientZ("gradient w.r.t nodal coordinates", mNumSpatialDims * mNumNodes);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tGradientZ);
    return tGradientZ;
}

template <typename PhysicsType>
std::string TimeIntegratedStateAverageFunction<PhysicsType>::name() const
{
    return std::string{};
}
}  // namespace plato::parabolic

#endif
