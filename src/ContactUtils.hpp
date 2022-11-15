#pragma once

#include "ContactPair.hpp"
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

std::vector<ContactPair> parse_contact
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh);

ContactPair parse_contact_pair
(const Teuchos::ParameterList & aParams,
 Plato::Mesh                    aMesh);

void parse_contact_penalty
(const Teuchos::ParameterList & aParams,
       ContactPair            & aContactPair);

Plato::SpatialDomain get_domain
(const std::string                       & aDomainName,
 const std::vector<Plato::SpatialDomain> & aDomains);

Plato::ScalarMultiVector compute_node_locations
(Plato::Mesh                                             aMesh,
 const Plato::OrdinalVectorT<const Plato::OrdinalType> & aNodes);

Plato::ScalarMultiVector map_node_locations
(const Plato::ScalarMultiVector      & aLocations,
 const Teuchos::Array<Plato::Scalar> & aTranslation);

Plato::OrdinalVector global_local_child_node_ord_map
(const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildNodes,
       Plato::OrdinalType                                aNumMeshNodes);

Plato::OrdinalVector convert_to_elementwise_map
(const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildElements,
 const Plato::OrdinalVectorT<const Plato::OrdinalType> & aChildFaceLocalNodes,
 const Plato::OrdinalVector                            & aMap,
       Plato::Mesh                                       aMesh,
       Plato::OrdinalType                                aNumNodesPerFace);

template<typename ElementType>
void set_parent_data_for_surface
(ContactSurface                      & aSurface,
 const Teuchos::Array<Plato::Scalar> & aTranslation,
 const Plato::SpatialModel           & aSpatialModel)
{
    auto tChildNodes = aSurface.childNodes();
    auto tGlobalLocalChildNodeOrdMap = global_local_child_node_ord_map(tChildNodes, aSpatialModel.Mesh->NumNodes());
    auto tElementWiseChildNodeOrdMap = convert_to_elementwise_map(aSurface.childElements(), aSurface.childFaceLocalNodes(), tGlobalLocalChildNodeOrdMap, aSpatialModel.Mesh, ElementType::mNumNodesPerFace);
    
    auto tChildLocations = compute_node_locations(aSpatialModel.Mesh, tChildNodes);
    auto tMappedChildLocations = map_node_locations(tChildLocations, aTranslation);
    Plato::SpatialDomain tDomain = get_domain(aSurface.parentBlock(), aSpatialModel.Domains);

    Plato::OrdinalVector tParentElements("parent elements", tChildNodes.size());
    Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
    (aSpatialModel.Mesh, tDomain.cellOrdinals(), tChildLocations, tMappedChildLocations, tParentElements);

    aSurface.addParentData(tParentElements, tElementWiseChildNodeOrdMap, tMappedChildLocations);
}

template<typename ElementType>
void set_parent_data_for_pairs
(std::vector<ContactPair>  & aPairs,
 const Plato::SpatialModel & aSpatialModel)
{
    for (auto & tPair : aPairs)
    {
        set_parent_data_for_surface<ElementType>(tPair.surfaceA, tPair.initialGap, aSpatialModel);

        auto tScaledGap = scale_initial_gap(tPair.initialGap, -1.0);
        set_parent_data_for_surface<ElementType>(tPair.surfaceB, tScaledGap, aSpatialModel);
    }
}

template<typename ElementType>
class ApplyContactPenalty : public ElementType
{
private:
    using ElementType::mNumSpatialDims;

public:
    ApplyContactPenalty(const Teuchos::Array<Plato::Scalar> & aPenaltyValue)
    {
        for(Plato::OrdinalType iDim=0; iDim < mNumSpatialDims; iDim++)
            mPenaltyValue(iDim) = aPenaltyValue[iDim];
    }

    template<typename StateType, typename ResultType>
    void
    operator()
    (const Plato::ScalarMultiVectorT<StateType>  & aState,
           Plato::ScalarMultiVectorT<ResultType> & aResult) const
    {
        auto tNumCells = aState.extent(0);
        
        auto tPenaltyValue = mPenaltyValue;
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumCells), KOKKOS_LAMBDA(Plato::OrdinalType iCellOrdinal)
        {
            for(Plato::OrdinalType iDim = 0; iDim < ElementType::mNumSpatialDims; iDim++)
                aResult(iCellOrdinal, iDim) = tPenaltyValue(iDim) * aState(iCellOrdinal, iDim);
        }, "apply contact penalization and projection");

    }

private:
    Plato::Array<mNumSpatialDims, Plato::Scalar> mPenaltyValue;

};

}

}
