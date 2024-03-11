#pragma once

#include "Hex8.hpp"
#include "Quad9.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Hex27 Element
*/
/******************************************************************************/
class Hex27
{
  public:

    using Face = Plato::Quad9;
    using C1 = Plato::Hex8;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 27;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 9;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 27;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    [[nodiscard]] static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        constexpr Plato::Scalar wc = 0.17146776406035665885063679070299; // 125/729
        constexpr Plato::Scalar we = 0.27434842249657065416101886512479; // 200/729
        constexpr Plato::Scalar wf = 0.43895747599451301335093944544496; // 320/729
        constexpr Plato::Scalar wb = 0.70233196159122079915704262020881; // 512/729

        return Plato::Array<mNumGaussPoints>{
            wc, we, wc, we, wf, we, wc, we, wc,
            we, wf, we, wf, wb, wf, we, wf, we,
            wc, we, wc, we, wf, we, wc, we, wc
        };
    }

    [[nodiscard]] static constexpr Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        constexpr Plato::Scalar p = 0.77459666924148340427791481488384; // sqrt(3.0/5.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>{
            -p, -p, -p,    0, -p, -p,    p, -p, -p,
            -p,  0, -p,    0,  0, -p,    p,  0, -p,
            -p,  p, -p,    0,  p, -p,    p,  p, -p,
            -p, -p,  0,    0, -p,  0,    p, -p,  0,
            -p,  0,  0,    0,  0,  0,    p,  0,  0,
            -p,  p,  0,    0,  p,  0,    p,  p,  0,
            -p, -p,  p,    0, -p,  p,    p, -p,  p,
            -p,  0,  p,    0,  0,  p,    p,  0,  p,
            -p,  p,  p,    0,  p,  p,    p,  p,  p
        };
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Array<mNumNodesPerCell>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto z=aCubPoint(2);

        const auto xyz=x*y*z;
        const auto yz=y*z;
        const auto xy=x*y;
        const auto xz=x*z;

        return Plato::Array<mNumNodesPerCell>
            {-(1-x)*(1-y)*(1-z)*xyz/8.0,
              (1+x)*(1-y)*(1-z)*xyz/8.0,
             -(1+x)*(1+y)*(1-z)*xyz/8.0,
              (1-x)*(1+y)*(1-z)*xyz/8.0,
              (1-x)*(1-y)*(1+z)*xyz/8.0,
             -(1+x)*(1-y)*(1+z)*xyz/8.0,
              (1+x)*(1+y)*(1+z)*xyz/8.0,
             -(1-x)*(1+y)*(1+z)*xyz/8.0,

              (1-x)*(1+x)*(1-y)*(1-z)*yz/4.0,
             -(1+x)*(1-y)*(1+y)*(1-z)*xz/4.0,
             -(1-x)*(1+x)*(1+y)*(1-z)*yz/4.0,
              (1-x)*(1-y)*(1+y)*(1-z)*xz/4.0,

              (1-x)*(1-y)*(1-z)*(1+z)*xy/4.0,
             -(1+x)*(1-y)*(1-z)*(1+z)*xy/4.0,
              (1+x)*(1+y)*(1-z)*(1+z)*xy/4.0,
             -(1-x)*(1+y)*(1-z)*(1+z)*xy/4.0,

             -(1-x)*(1+x)*(1-y)*(1+z)*yz/4.0,
              (1+x)*(1-y)*(1+y)*(1+z)*xz/4.0,
              (1-x)*(1+x)*(1+y)*(1+z)*yz/4.0,
             -(1-x)*(1-y)*(1+y)*(1+z)*xz/4.0,

              (1-x)*(1+x)*(1-y)*(1+y)*(1-z)*(1+z),

             -(1-x)*(1+x)*(1-y)*(1+y)*(1-z)*z/2.0,
              (1-x)*(1+x)*(1-y)*(1+y)*(1+z)*z/2.0,

             -(1-x)*x*(1-y)*(1+y)*(1-z)*(1+z)/2.0,
              (1+x)*x*(1-y)*(1+y)*(1-z)*(1+z)/2.0,

             -(1-x)*(1+x)*(1-y)*y*(1-z)*(1+z)/2.0,
              (1-x)*(1+x)*(1+y)*y*(1-z)*(1+z)/2.0};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto z=aCubPoint(2);
        const auto xyz=x*y*z;
        const auto yz=y*z;
        const auto xy=x*y;
        const auto xz=x*z;

        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            {-(1-2*x)*(1-y)*(1-z)*yz/8.0,          -(1-x)*(1-2*y)*(1-z)*xz/8.0,          -(1-x)*(1-y)*(1-2*z)*xy/8.0,
              (1+2*x)*(1-y)*(1-z)*yz/8.0,           (1+x)*(1-2*y)*(1-z)*xz/8.0,           (1+x)*(1-y)*(1-2*z)*xy/8.0,
             -(1+2*x)*(1+y)*(1-z)*yz/8.0,          -(1+x)*(1+2*y)*(1-z)*xz/8.0,          -(1+x)*(1+y)*(1-2*z)*xy/8.0,
              (1-2*x)*(1+y)*(1-z)*yz/8.0,           (1-x)*(1+2*y)*(1-z)*xz/8.0,           (1-x)*(1+y)*(1-2*z)*xy/8.0,
              (1-2*x)*(1-y)*(1+z)*yz/8.0,           (1-x)*(1-2*y)*(1+z)*xz/8.0,           (1-x)*(1-y)*(1+2*z)*xy/8.0,
             -(1+2*x)*(1-y)*(1+z)*yz/8.0,          -(1+x)*(1-2*y)*(1+z)*xz/8.0,          -(1+x)*(1-y)*(1+2*z)*xy/8.0,
              (1+2*x)*(1+y)*(1+z)*yz/8.0,           (1+x)*(1+2*y)*(1+z)*xz/8.0,           (1+x)*(1+y)*(1+2*z)*xy/8.0,
             -(1-2*x)*(1+y)*(1+z)*yz/8.0,          -(1-x)*(1+2*y)*(1+z)*xz/8.0,          -(1-x)*(1+y)*(1+2*z)*xy/8.0,
             -(1-y)*(1-z)*xyz/2.0,                  (1-x)*(1+x)*(1-2*y)*(1-z)*z/4.0,      (1-x)*(1+x)*(1-y)*(1-2*z)*y/4.0,
             -(1+2*x)*(1-y)*(1+y)*(1-z)*z/4.0,      (1+x)*(1-z)*xyz/2.0,                 -(1+x)*(1-y)*(1+y)*(1-2*z)*x/4.0,
              (1+y)*(1-z)*xyz/2.0,                 -(1-x)*(1+x)*(1+2*y)*(1-z)*z/4.0,     -(1-x)*(1+x)*(1+y)*(1-2*z)*y/4.0,
              (1-2*x)*(1-y)*(1+y)*(1-z)*z/4.0,     -(1-x)*(1-z)*xyz/2.0,                  (1-x)*(1-y)*(1+y)*(1-2*z)*x/4.0,
              (1-2*x)*(1-y)*(1-z)*(1+z)*y/4.0,      (1-x)*(1-2*y)*(1-z)*(1+z)*x/4.0,     -(1-x)*(1-y)*xyz/2.0,
             -(1+2*x)*(1-y)*(1-z)*(1+z)*y/4.0,     -(1+x)*(1-2*y)*(1-z)*(1+z)*x/4.0,      (1+x)*(1-y)*xyz/2.0,
              (1+2*x)*(1+y)*(1-z)*(1+z)*y/4.0,      (1+x)*(1+2*y)*(1-z)*(1+z)*x/4.0,     -(1+x)*(1+y)*xyz/2.0,
             -(1-2*x)*(1+y)*(1-z)*(1+z)*y/4.0,     -(1-x)*(1+2*y)*(1-z)*(1+z)*x/4.0,      (1-x)*(1+y)*xyz/2.0,
              (1-y)*(1+z)*xyz/2.0,                 -(1-x)*(1+x)*(1-2*y)*(1+z)*z/4.0,     -(1-x)*(1+x)*(1-y)*(1+2*z)*y/4.0,
              (1+2*x)*(1-y)*(1+y)*(1+z)*z/4.0,     -(1+x)*(1+z)*xyz/2.0,                  (1+x)*(1-y)*(1+y)*(1+2*z)*x/4.0,
             -(1+y)*(1+z)*xyz/2.0,                  (1-x)*(1+x)*(1+2*y)*(1+z)*z/4.0,      (1-x)*(1+x)*(1+y)*(1+2*z)*y/4.0,
             -(1-2*x)*(1-y)*(1+y)*(1+z)*z/4.0,      (1-x)*(1+z)*xyz/2.0,                 -(1-x)*(1-y)*(1+y)*(1+2*z)*x/4.0,
             -2*x*(1-y)*(1+y)*(1-z)*(1+z),         -2*y*(1-x)*(1+x)*(1-z)*(1+z),         -2*z*(1-x)*(1+x)*(1-y)*(1+y),
              (1-y)*(1+y)*(1-z)*xz,                 (1-x)*(1+x)*(1-z)*yz,                -(1-x)*(1+x)*(1-y)*(1+y)*(1-2*z)/2.0,
             -(1-y)*(1+y)*(1+z)*xz,                -(1-x)*(1+x)*(1+z)*yz,                 (1-x)*(1+x)*(1-y)*(1+y)*(1+2*z)/2.0,
             -(1-2*x)*(1-y)*(1+y)*(1-z)*(1+z)/2.0,  (1-x)*(1-z)*(1+z)*xy,                 (1-x)*(1-y)*(1+y)*xz,
              (1+2*x)*(1-y)*(1+y)*(1-z)*(1+z)/2.0, -(1+x)*(1-z)*(1+z)*xy,                -(1+x)*(1-y)*(1+y)*xz,
              (1-y)*(1-z)*(1+z)*xy,                -(1-x)*(1+x)*(1-2*y)*(1-z)*(1+z)/2.0,  (1-x)*(1+x)*(1-y)*yz,
             -(1+y)*(1-z)*(1+z)*xy,                 (1-x)*(1+x)*(1+2*y)*(1-z)*(1+z)/2.0, -(1-x)*(1+x)*(1+y)*yz};
    }
};

} // end namespace Plato
