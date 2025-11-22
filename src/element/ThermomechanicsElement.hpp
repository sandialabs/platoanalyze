#pragma once

#include "element/ElementBase.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for thermomechanics element
 */
/******************************************************************************/
template <typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class ThermomechanicsElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
   public:
    using TopoElementTypeT::mNumGaussPoints;
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumSpatialDims;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumVoigtTerms =
        (mNumSpatialDims == 3) ? 6 : ((mNumSpatialDims == 2) ? 3 : (((mNumSpatialDims == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mTDofOffset = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerNode = mNumSpatialDims + 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;

    static constexpr Plato::OrdinalType mNumLocalStatesPerGP = 0;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = mNumLocalStatesPerGP * mNumGaussPoints;
};

}  // namespace Plato
