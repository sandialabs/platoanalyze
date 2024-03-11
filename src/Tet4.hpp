#pragma once

#include "Tri3.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tet4 Element
*/
/******************************************************************************/
class Tet4
{
  public:

    using Face = Plato::Tri3;
    using C1 = Plato::Tet4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 4;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 3;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 4;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights() 
    {
        return Plato::Array<mNumGaussPoints>
                 ({Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0});
    }

    static constexpr Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<mNumGaussPoints, mNumSpatialDims>({
            0.585410196624969, 0.138196601125011, 0.138196601125011,
            0.138196601125011, 0.585410196624969, 0.138196601125011,
            0.138196601125011, 0.138196601125011, 0.585410196624969,
            0.138196601125011, 0.138196601125011, 0.138196601125011
        });
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Array<mNumNodesPerCell>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto z=aCubPoint(2);

        return Plato::Array<mNumNodesPerCell>
            {Plato::Scalar(1) - x - y - z,
             x,
             y,
             z};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            {-1, -1, -1,
              1,  0,  0,
              0,  1,  0,
              0,  0,  1};
    }
};

} // end namespace Plato
