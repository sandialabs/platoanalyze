#pragma once

#include <Teuchos_ParameterList.hpp>
#include <string>
#include <vector>

#include "domain/SpatialModel.hpp"
#include "domain/contact/ContactPair.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "mesh/Plato_MeshMap.hpp"

namespace Plato
{

namespace Contact
{

std::vector<ContactPair> parse_contact(const Teuchos::ParameterList& aParams, Plato::Mesh aMesh);

ContactPair parse_contact_pair(const Teuchos::ParameterList& aParams, Plato::Mesh aMesh);

void parse_contact_penalty(const Teuchos::ParameterList& aParams, ContactPair& aContactPair);

plato::domain::SpatialDomain get_domain(const std::string& aDomainName,
                                        const std::vector<plato::domain::SpatialDomain>& aDomains);

Plato::ScalarMultiVector compute_node_locations(Plato::Mesh aMesh,
                                                const Plato::OrdinalVectorT<const Plato::OrdinalType>& aNodes);

Plato::ScalarMultiVector map_node_locations(const Plato::ScalarMultiVector& aLocations,
                                            const Teuchos::Array<Plato::Scalar>& aTranslation);

Plato::OrdinalVector global_local_child_node_ord_map(const Plato::OrdinalVectorT<const Plato::OrdinalType>& aChildNodes,
                                                     Plato::OrdinalType aNumMeshNodes);

Plato::OrdinalVector convert_to_elementwise_map(
    const Plato::OrdinalVectorT<const Plato::OrdinalType>& aChildElements,
    const Plato::OrdinalVectorT<const Plato::OrdinalType>& aChildFaceLocalNodes,
    const Plato::OrdinalVector& aMap,
    Plato::Mesh aMesh,
    Plato::OrdinalType aNumNodesPerFace);

void check_for_missing_parent_elements(const Plato::OrdinalVector& aParentElements);

template <typename ElementType>
void set_parent_data_for_surface(ContactSurface& aSurface,
                                 const Teuchos::Array<Plato::Scalar>& aTranslation,
                                 const plato::domain::SpatialModel& aSpatialModel,
                                 Plato::Scalar aSearchTolerance)
{
    auto tChildNodes = aSurface.childNodes();
    auto tGlobalLocalChildNodeOrdMap = global_local_child_node_ord_map(tChildNodes, aSpatialModel.mMesh->NumNodes());
    auto tElementWiseChildNodeOrdMap =
        convert_to_elementwise_map(aSurface.childElements(), aSurface.childFaceLocalNodes(),
                                   tGlobalLocalChildNodeOrdMap, aSpatialModel.mMesh, ElementType::mNumNodesPerFace);

    auto tChildLocations = compute_node_locations(aSpatialModel.mMesh, tChildNodes);
    auto tMappedChildLocations = map_node_locations(tChildLocations, aTranslation);
    plato::domain::SpatialDomain tDomain = get_domain(aSurface.parentBlock(), aSpatialModel.mDomains);

    Plato::OrdinalVector tParentElements("parent elements", tChildNodes.size());
    if (aSearchTolerance > 0)
    {
        Plato::Geometry::findParentElements<ElementType, Plato::Scalar>(aSpatialModel.mMesh, tDomain.cellOrdinals(),
                                                                        tChildLocations, tMappedChildLocations,
                                                                        tParentElements, aSearchTolerance);
    }
    else
    {
        Plato::Geometry::findParentElements<ElementType, Plato::Scalar>(
            aSpatialModel.mMesh, tDomain.cellOrdinals(), tChildLocations, tMappedChildLocations, tParentElements);
    }

    check_for_missing_parent_elements(tParentElements);

    aSurface.addParentData(tParentElements, tElementWiseChildNodeOrdMap, tMappedChildLocations);
}

template <typename ElementType>
void set_parent_data_for_pairs(std::vector<ContactPair>& aPairs, const plato::domain::SpatialModel& aSpatialModel)
{
    for (auto& tPair : aPairs)
    {
        set_parent_data_for_surface<ElementType>(tPair.surfaceA, tPair.initialGap, aSpatialModel,
                                                 tPair.searchTolerance);

        auto tScaledGap = scale_initial_gap(tPair.initialGap, -1.0);
        set_parent_data_for_surface<ElementType>(tPair.surfaceB, tScaledGap, aSpatialModel, tPair.searchTolerance);
    }
}

}  // namespace Contact

}  // namespace Plato
