#ifndef PLATO_ELEMENT_GEOMETRICELEMENT_H
#define PLATO_ELEMENT_GEOMETRICELEMENT_H

#include "element/ElementBase.hpp"

namespace plato::element
{
/// @brief class containing element information for geometric operations
template <typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class GeometricElement : public TopoElementTypeT, public Plato::ElementBase<TopoElementTypeT>
{
   public:
    using TopoElementType = TopoElementTypeT;

    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumDofsPerNode = 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
    static constexpr Plato::OrdinalType mNumLocalStatesPerGP = 0;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};
}  // namespace plato::element

#endif
