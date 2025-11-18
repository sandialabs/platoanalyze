#ifndef PLATO_CONTACT_EVALUATEELEMENTCONTACTFORCES_H
#define PLATO_CONTACT_EVALUATEELEMENTCONTACTFORCES_H

#include "domain/SpatialModel.hpp"
#include "domain/WorksetBase.hpp"
#include "domain/contact/ContactForceFactory.hpp"
#include "domain/contact/IntegrateContactForce.hpp"
#include "domain/contact/SurfaceDisplacementFactory.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace Plato::Contact
{
template <typename EvaluationType>
[[nodiscard]] auto element_contact_force_contribution(const Plato::SpatialModel& aSpatialModel,
                                                      const Plato::ScalarVector& aState,
                                                      Plato::Scalar aTimeStep = 0.0)
    -> Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType>
{
    using ElementType = typename EvaluationType::ElementType;

    using ConfigScalar = typename EvaluationType::ConfigScalarType;
    using StateScalar = typename EvaluationType::StateScalarType;
    using ResultScalar = typename EvaluationType::ResultScalarType;

    const auto tNumCells = aSpatialModel.Mesh->NumElements();

    const Plato::WorksetBase<ElementType> tWorksetBase(aSpatialModel.Mesh);

    // workset config
    Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, ElementType::mNumNodesPerCell,
                                                  ElementType::mNumSpatialDims);
    tWorksetBase.worksetConfig(tConfigWS);

    // workset state
    Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(aState, tStateWS);

    Plato::ScalarMultiVectorT<ResultScalar> tElementContactForceValues("", tNumCells, ElementType::mNumDofsPerCell);

    const auto tPairs = aSpatialModel.contactPairs();
    const SurfaceDisplacementFactory<EvaluationType> tSurfaceDisplacementFactory;
    const ContactForceFactory<EvaluationType> tContactForceFactory;

    for (const auto tPair : tPairs)
    {
        const auto tComputeContactForce = tContactForceFactory.create(tPair.penaltyType, tPair.penaltyValue);

        const auto tSideSetA = tPair.surfaceA.childSideSet();
        const auto tComputeChildSurfaceDispA = tSurfaceDisplacementFactory.createChildContribution(tPair.surfaceA);
        const IntegrateContactForce<EvaluationType> tIntegrateContactForceChildA(
            aSpatialModel, tSideSetA, tComputeChildSurfaceDispA, tComputeContactForce);
        tIntegrateContactForceChildA(tStateWS, tConfigWS, tElementContactForceValues, aTimeStep);

        auto tComputeParentSurfaceDispA =
            tSurfaceDisplacementFactory.createParentContribution(tPair.surfaceA, aSpatialModel.Mesh, -1.0);
        for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
        {
            tComputeParentSurfaceDispA->setChildNode(iChildNode);
            const IntegrateContactForce<EvaluationType> tIntegrateContactForceParentA(
                aSpatialModel, tSideSetA, tComputeParentSurfaceDispA, tComputeContactForce);
            tIntegrateContactForceParentA(tStateWS, tConfigWS, tElementContactForceValues, aTimeStep);
        }

        const auto tSideSetB = tPair.surfaceB.childSideSet();
        const auto tComputeChildSurfaceDispB = tSurfaceDisplacementFactory.createChildContribution(tPair.surfaceB);
        const IntegrateContactForce<EvaluationType> tIntegrateContactForceChildB(
            aSpatialModel, tSideSetB, tComputeChildSurfaceDispB, tComputeContactForce);
        tIntegrateContactForceChildB(tStateWS, tConfigWS, tElementContactForceValues, aTimeStep);

        auto tComputeParentSurfaceDispB =
            tSurfaceDisplacementFactory.createParentContribution(tPair.surfaceB, aSpatialModel.Mesh, -1.0);
        for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
        {
            tComputeParentSurfaceDispB->setChildNode(iChildNode);
            const IntegrateContactForce<EvaluationType> tIntegrateContactForceParentB(
                aSpatialModel, tSideSetB, tComputeParentSurfaceDispB, tComputeContactForce);
            tIntegrateContactForceParentB(tStateWS, tConfigWS, tElementContactForceValues, aTimeStep);
        }
    }

    return tElementContactForceValues;
}

template <typename EvaluationType, typename EntryOrdinalType>
void assemble_contact_force_nonlocal_jacobian(const Plato::SpatialModel& aSpatialModel,
                                              Teuchos::RCP<Plato::CrsMatrixType> aInputMatrix,
                                              const EntryOrdinalType& aEntryOrdinal,
                                              const Plato::ScalarVector& aState,
                                              Plato::Scalar aTimeStep = 0.0)
{
    using ElementType = typename EvaluationType::ElementType;

    using ConfigScalar = typename EvaluationType::ConfigScalarType;
    using StateScalar = typename EvaluationType::StateScalarType;
    using ResultScalar = typename EvaluationType::ResultScalarType;

    const auto tNumCells = aSpatialModel.Mesh->NumElements();

    const Plato::WorksetBase<ElementType> tWorksetBase(aSpatialModel.Mesh);

    // workset config
    Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, ElementType::mNumNodesPerCell,
                                                  ElementType::mNumSpatialDims);
    tWorksetBase.worksetConfig(tConfigWS);

    // workset state
    Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, ElementType::mNumDofsPerCell);
    tWorksetBase.worksetState(aState, tStateWS);

    const auto tPairs = aSpatialModel.contactPairs();
    const SurfaceDisplacementFactory<EvaluationType> tSurfaceDisplacementFactory;
    const ContactForceFactory<EvaluationType> tContactForceFactory;

    auto tMatEntries = aInputMatrix->entries();

    for (const auto tPair : tPairs)
    {
        const auto tComputeContactForce = tContactForceFactory.create(tPair.penaltyType, tPair.penaltyValue);

        const auto tSideSetA = tPair.surfaceA.childSideSet();
        const auto tChildCellsA = tPair.surfaceA.childElements();
        const auto tParentCellsA = tPair.surfaceA.parentElements();
        const auto tElementWiseChildMapA = tPair.surfaceA.elementWiseChildMap();
        const auto tChildFaceLocalNodesA = tPair.surfaceA.childFaceLocalNodes();

        const auto tComputeChildSurfaceDispA = tSurfaceDisplacementFactory.createChildContribution(tPair.surfaceA);
        Plato::ScalarMultiVectorT<ResultScalar> tResultA("Results side A", tNumCells, ElementType::mNumDofsPerCell);
        const IntegrateContactForce<EvaluationType> tIntegrateContactForceChildA(
            aSpatialModel, tSideSetA, tComputeChildSurfaceDispA, tComputeContactForce);
        tIntegrateContactForceChildA(tStateWS, tConfigWS, tResultA, aTimeStep);

        tWorksetBase.assembleJacobianFad(ElementType::mNumDofsPerCell, ElementType::mNumDofsPerCell, aEntryOrdinal,
                                         tResultA, tMatEntries);

        auto tComputeParentSurfaceDispA =
            tSurfaceDisplacementFactory.createParentContribution(tPair.surfaceA, aSpatialModel.Mesh, -1.0);
        for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
        {
            tComputeParentSurfaceDispA->setChildNode(iChildNode);
            Plato::ScalarMultiVectorT<ResultScalar> tResultA("Results side A", tNumCells, ElementType::mNumDofsPerCell);
            const IntegrateContactForce<EvaluationType> tIntegrateContactForceParentA(
                aSpatialModel, tSideSetA, tComputeParentSurfaceDispA, tComputeContactForce);
            tIntegrateContactForceParentA(tStateWS, tConfigWS, tResultA, aTimeStep);

            tWorksetBase.assembleJacobianFad(ElementType::mNumDofsPerCell, tChildCellsA, tParentCellsA,
                                             tElementWiseChildMapA, tChildFaceLocalNodesA, iChildNode, aEntryOrdinal,
                                             tResultA, tMatEntries);
        }

        const auto tSideSetB = tPair.surfaceB.childSideSet();
        const auto tChildCellsB = tPair.surfaceB.childElements();
        const auto tParentCellsB = tPair.surfaceB.parentElements();
        const auto tElementWiseChildMapB = tPair.surfaceB.elementWiseChildMap();
        const auto tChildFaceLocalNodesB = tPair.surfaceB.childFaceLocalNodes();

        const auto tComputeChildSurfaceDispB = tSurfaceDisplacementFactory.createChildContribution(tPair.surfaceB);
        Plato::ScalarMultiVectorT<ResultScalar> tResultB("Results side B", tNumCells, ElementType::mNumDofsPerCell);
        const IntegrateContactForce<EvaluationType> tIntegrateContactForceChildB(
            aSpatialModel, tSideSetB, tComputeChildSurfaceDispB, tComputeContactForce);
        tIntegrateContactForceChildB(tStateWS, tConfigWS, tResultB, aTimeStep);

        tWorksetBase.assembleJacobianFad(ElementType::mNumDofsPerCell, ElementType::mNumDofsPerCell, aEntryOrdinal,
                                         tResultB, tMatEntries);

        auto tComputeParentSurfaceDispB =
            tSurfaceDisplacementFactory.createParentContribution(tPair.surfaceB, aSpatialModel.Mesh, -1.0);
        for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
        {
            tComputeParentSurfaceDispB->setChildNode(iChildNode);
            Plato::ScalarMultiVectorT<ResultScalar> tResultB("Results side B", tNumCells, ElementType::mNumDofsPerCell);
            const IntegrateContactForce<EvaluationType> tIntegrateContactForceParentB(
                aSpatialModel, tSideSetB, tComputeParentSurfaceDispB, tComputeContactForce);
            tIntegrateContactForceParentB(tStateWS, tConfigWS, tResultB, aTimeStep);

            tWorksetBase.assembleJacobianFad(ElementType::mNumDofsPerCell, tChildCellsB, tParentCellsB,
                                             tElementWiseChildMapB, tChildFaceLocalNodesB, iChildNode, aEntryOrdinal,
                                             tResultB, tMatEntries);
        }
    }
}
}  // namespace Plato::Contact

#endif
