#pragma once

#include "Tri6.hpp"
#include "Tet4.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tet10 Element
*/
/******************************************************************************/
class Tet10
{
  public:

    using Face = Plato::Tri6;
    using C1 = Plato::Tet4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 10;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 6;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 4;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        return Plato::Array<mNumGaussPoints>
                 ({Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0, Plato::Scalar(1.0)/24.0});
    }

    static constexpr Plato::Matrix<mNumGaussPoints, mNumSpatialDims>
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

        const auto tCon = (x+y+z-1.0);

        return Plato::Array<mNumNodesPerCell>
            { tCon*(2.0*x+2.0*y+2.0*z-1),
              x*(2.0*x-1.0),
              y*(2.0*y-1.0),
              z*(2.0*z-1.0),
             -4.0*x*tCon,
              4.0*x*y,
             -4.0*y*tCon,
             -4.0*z*tCon,
              4.0*x*z,
              4.0*y*z};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto z=aCubPoint(2);

        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            {(x+y+z)*4.0-3.0    , (x+y+z)*4.0-3.0    , (x+y+z)*4.0-3.0,
             x*4.0-1.0          , 0.0                , 0.0,
             0.0                , y*4.0-1.0          , 0.0,
             0.0                , 0.0                , z*4.0-1.0,
             -(x*2.0+y+z-1)*4.0 , -x*4.0             , -x*4.0,
             y*4.0              , x*4.0              , 0.0,
             -y*4.0             , -(x+2.0*y+z-1)*4.0 , -y*4.0,
             -z*4.0             , -z*4.0             , -(x+y+z*2.0-1)*4.,
             z*4.0              , 0.0                , x*4.,
             0.0                , z*4.0              , y*4.};
    }
};

} // end namespace Plato
