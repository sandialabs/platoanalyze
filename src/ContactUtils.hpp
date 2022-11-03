#pragma once

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "SpatialModel.hpp"
#include "Plato_MeshMap.hpp"

#include <Teuchos_ParameterList.hpp>
#include <string>
#include <vector>

namespace Plato
{

namespace Contact
{

struct ContactPair
{
    std::string childSideSetA;
    std::string childSideSetB;

    Plato::OrdinalVectorT<const Plato::OrdinalType> childNodesA;
    Plato::OrdinalVectorT<const Plato::OrdinalType> childNodesB;

    Plato::OrdinalVectorT<const Plato::OrdinalType> childElementsA;
    Plato::OrdinalVectorT<const Plato::OrdinalType> childElementsB;

    Plato::OrdinalVectorT<const Plato::OrdinalType> childFaceLocalNodesA;
    Plato::OrdinalVectorT<const Plato::OrdinalType> childFaceLocalNodesB;

    std::string parentBlockA;
    std::string parentBlockB;

    Teuchos::Array<Plato::Scalar> initialGap;
};

ContactPair parse_contact_pair
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh);

std::vector<ContactPair> parse_contact
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh);

Plato::SpatialDomain get_domain
(const std::string                       & aDomainName,
 const std::vector<Plato::SpatialDomain> & aDomains);

Plato::ScalarMultiVector compute_node_locations
(Plato::Mesh                                             aMesh,
 const Plato::OrdinalVectorT<const Plato::OrdinalType> & aNodes);

Plato::ScalarMultiVector map_node_locations
(const Plato::ScalarMultiVector      & aLocations,
 const Teuchos::Array<Plato::Scalar> & aTranslation,
 Plato::Scalar                         aScale = 1.0);

Plato::OrdinalVector global_local_child_node_ord_map
(const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildNodes,
       Plato::OrdinalType                                aNumMeshNodes);

Plato::OrdinalVector convert_to_elementwise_map
(const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildElements,
 const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildFaceLocalNodes,
 const Plato::OrdinalVector                            & aMap,
       Plato::Mesh                                       aMesh,
       Plato::OrdinalType                                aNumNodesPerFace);

Plato::OrdinalType count_total_child_nodes(const std::vector<ContactPair> & aPairs);

template<typename ElementType>
void populate_full_contact_arrays
(const std::vector<ContactPair> & aPairs,
 const Plato::SpatialModel      & aSpatialModel,
       Plato::OrdinalVector     & aChildNodes,
       Plato::OrdinalVector     & aParentElements)
{
    Plato::OrdinalType tOffset(0);
    for (auto tPair : aPairs)
    {
        // side A parents
        Plato::OrdinalType tNumNodes = tPair.childNodesA.size();
        auto tChildLocations       = compute_node_locations(aSpatialModel.Mesh, tPair.childNodesA);
        auto tMappedChildLocations = map_node_locations(tChildLocations, tPair.initialGap);
        Plato::SpatialDomain tDomain = get_domain(tPair.parentBlockB, aSpatialModel.Domains);

        Plato::OrdinalVector tParentElementsA("parent elements", tNumNodes);
        Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
        (aSpatialModel.Mesh, tDomain.cellOrdinals(), tChildLocations, tMappedChildLocations, tParentElementsA);

        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
        {
            aChildNodes(tOffset + nodeOrdinal) = tPair.childNodesA(nodeOrdinal);
            aParentElements(tOffset + nodeOrdinal) = tParentElementsA(nodeOrdinal);
        }, "store parent elements");
        tOffset += tNumNodes;

        // side B parents
        tNumNodes = tPair.childNodesB.size();
        tChildLocations       = compute_node_locations(aSpatialModel.Mesh, tPair.childNodesB);
        tMappedChildLocations = map_node_locations(tChildLocations, tPair.initialGap, -1.0);
        tDomain = get_domain(tPair.parentBlockA, aSpatialModel.Domains);

        Plato::OrdinalVector tParentElementsB("parent elements", tNumNodes);
        Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
        (aSpatialModel.Mesh, tDomain.cellOrdinals(), tChildLocations, tMappedChildLocations, tParentElementsB);

        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType nodeOrdinal)
        {
            aChildNodes(tOffset + nodeOrdinal) = tPair.childNodesB(nodeOrdinal);
            aParentElements(tOffset + nodeOrdinal) = tParentElementsB(nodeOrdinal);
        }, "store parent elements");
        tOffset += tNumNodes;
    }
}

void check_for_repeated_child_nodes
(const Plato::OrdinalVector & aChildNodes,
       Plato::Mesh            aMesh);

}

}
