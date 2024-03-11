#pragma once

#include "ElementBase.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for thermomechanics element
*/
/******************************************************************************/
template<typename TopoElementTypeT, Plato::OrdinalType NumControls = 1>
class ThermomechanicsElement : public TopoElementTypeT, public ElementBase<TopoElementTypeT>
{
  public:
    using TopoElementTypeT::mNumNodesPerCell;
    using TopoElementTypeT::mNumSpatialDims;
    using TopoElementTypeT::mNumGaussPoints;

    using TopoElementType = TopoElementTypeT;

    static constexpr Plato::OrdinalType mNumVoigtTerms   = (mNumSpatialDims == 3) ? 6 :
                                                          ((mNumSpatialDims == 2) ? 3 :
                                                         (((mNumSpatialDims == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mTDofOffset      = mNumSpatialDims;
    static constexpr Plato::OrdinalType mNumDofsPerNode  = mNumSpatialDims + 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;

    static constexpr Plato::OrdinalType mNumLocalStatesPerGP = 0;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = mNumLocalStatesPerGP*mNumGaussPoints;


};

#ifdef NOPE
/******************************************************************************/
/*! Base class for simplex-based two-field thermomechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexStabilizedThermomechanics : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumVoigtTerms =
            (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0)));

    // degree-of-freedom attributes
    //
    static constexpr Plato::OrdinalType mTDofOffset        = SpaceDim + 1;
    static constexpr Plato::OrdinalType mNumDofsPerNode    = SpaceDim + 2;
    static constexpr Plato::OrdinalType mPressureDofOffset = SpaceDim;
    static constexpr Plato::OrdinalType mNumDofsPerCell    = mNumDofsPerNode * mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumControl        = NumControls;

    // this physics can be used with VMS functionality in PA.  The
    // following defines the nodal state attributes required by VMS
    //
    static constexpr Plato::OrdinalType mNumNodeStatePerNode = SpaceDim;
    static constexpr Plato::OrdinalType mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
};
#endif


} // namespace Plato
