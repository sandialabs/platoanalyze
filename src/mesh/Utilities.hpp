#ifndef SRC_PLATO_MESH_UTILITIES_HPP_
#define SRC_PLATO_MESH_UTILITIES_HPP_

#include <typeinfo>

#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "utilities/Variables.hpp"

#ifdef USE_OMEGAH_MESH
#include <Omega_h_shape.hpp>
#endif

namespace Plato
{

inline void readNodeFields(Plato::MeshIO aReader,
                           Plato::OrdinalType aStepIndex,
                           Plato::FieldTags aFieldTags,
                           Plato::Variables& aVariables)
{
    auto tTags = aFieldTags.tags();
    for (auto& tTag : tTags)
    {
        auto tData = aReader->ReadNodeData(tTag, aStepIndex);
        auto tFieldName = aFieldTags.id(tTag);
        aVariables.vector(tFieldName, tData);
    }
}

/******************************************************************************/
/**
 * \tparam NumSpatialDims  number of spatial dimensions
 * \tparam NumNodesPerCell number of nodes per cell/element
 *
 * \fn Scalar calculate_element_size
 *
 * \brief Calculate characteristic element size
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aCells2Nodes map from cells to node ordinal
 * \param [in] aCoords      cell/element coordinates
 **********************************************************************************/
template <Plato::OrdinalType NumSpatialDims, Plato::OrdinalType NumNodesPerCell>
KOKKOS_INLINE_FUNCTION Plato::Scalar calculate_element_size(
    const Plato::OrdinalType& aCellOrdinal,
    const Plato::OrdinalVectorT<const Plato::OrdinalType>& aConnectivity,
    const Plato::OrdinalVectorT<const Plato::Scalar>& aCoordinates)
{
#ifdef USE_OMEGAH_MESH
    Omega_h::Few<Omega_h::Vector<NumSpatialDims>, NumNodesPerCell> tElemCoords;
    for (Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        const Plato::OrdinalType tVertexIndex = aConnectivity(aCellOrdinal * NumNodesPerCell + tNode);
        for (Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            tElemCoords[tNode][tDim] = aCoordinates(tVertexIndex * NumSpatialDims + tDim);
        }
    }
    auto tSphere = Omega_h::get_inball(tElemCoords);

    return (static_cast<Plato::Scalar>(2.0) * tSphere.r);
#else
    ANALYZE_THROWERR("Omega-h is disabled. calculate_element_size() is not available");
#endif
}
// function calculate_element_size
}  // namespace Plato
#endif
