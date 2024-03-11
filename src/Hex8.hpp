#pragma once

#include "Quad4.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Hex8 Element
*/
/******************************************************************************/
class Hex8
{
  public:

    using Face = Plato::Quad4;
    using C1 = Plato::Hex8;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 8;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 4;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 8;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    [[nodiscard]] static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        return Plato::Array<mNumGaussPoints>{
            Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0),
            Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0)
        };
    }

    [[nodiscard]] static constexpr Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        const Plato::Scalar sqt = 0.57735026918962584208117050366127; // sqrt(1.0/3.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>{
            -sqt, -sqt, -sqt,
             sqt, -sqt, -sqt,
             sqt,  sqt, -sqt,
            -sqt,  sqt, -sqt,
            -sqt, -sqt,  sqt,
             sqt, -sqt,  sqt,
             sqt,  sqt,  sqt,
            -sqt,  sqt,  sqt,
        };
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Array<mNumNodesPerCell>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto z=aCubPoint(2);

        return Plato::Array<mNumNodesPerCell>
            {(1-x)*(1-y)*(1-z)/8.0,
             (1+x)*(1-y)*(1-z)/8.0,
             (1+x)*(1+y)*(1-z)/8.0,
             (1-x)*(1+y)*(1-z)/8.0,
             (1-x)*(1-y)*(1+z)/8.0,
             (1+x)*(1-y)*(1+z)/8.0,
             (1+x)*(1+y)*(1+z)/8.0,
             (1-x)*(1+y)*(1+z)/8.0};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto z=aCubPoint(2);

        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            {-(1-y)*(1-z)/8.0, -(1-x)*(1-z)/8.0, -(1-x)*(1-y)/8.0,
              (1-y)*(1-z)/8.0, -(1+x)*(1-z)/8.0, -(1+x)*(1-y)/8.0,
              (1+y)*(1-z)/8.0,  (1+x)*(1-z)/8.0, -(1+x)*(1+y)/8.0,
             -(1+y)*(1-z)/8.0,  (1-x)*(1-z)/8.0, -(1-x)*(1+y)/8.0,
             -(1-y)*(1+z)/8.0, -(1-x)*(1+z)/8.0,  (1-x)*(1-y)/8.0,
              (1-y)*(1+z)/8.0, -(1+x)*(1+z)/8.0,  (1+x)*(1-y)/8.0,
              (1+y)*(1+z)/8.0,  (1+x)*(1+z)/8.0,  (1+x)*(1+y)/8.0,
             -(1+y)*(1+z)/8.0,  (1-x)*(1+z)/8.0,  (1-x)*(1+y)/8.0};
    }
};

} // end namespace Plato
