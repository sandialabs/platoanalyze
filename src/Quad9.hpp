#pragma once

#include "Quad4.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Quad9 Element
*/
/******************************************************************************/
class Quad9
{
  public:
    using C1 = Plato::Quad4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 9;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 3;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 9;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    [[nodiscard]] static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        constexpr Plato::Scalar wc = 0.30864197530864195817557060763647; // 25/81
        constexpr Plato::Scalar we = 0.49382716049382713308091297221836; // 40/81
        constexpr Plato::Scalar wf = 0.79012345679012341292946075554937; // 64/81

        return Plato::Array<mNumGaussPoints>{
            wc, we, wc, we, wf, we, wc, we, wc
        };
    }

    [[nodiscard]] static constexpr Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        constexpr Plato::Scalar p = 0.77459666924148340427791481488384; // sqrt(3.0/5.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>{
            -p, -p,   0, -p,   p, -p,
            -p,  0,   0,  0,   p,  0,
            -p,  p,   0,  p,   p,  p
        };
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Array<mNumNodesPerCell>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto xy = x*y;

        return Plato::Array<mNumNodesPerCell>
            { (1-x)*(1-y)*xy/4.0,
             -(1+x)*(1-y)*xy/4.0,
              (1+x)*(1+y)*xy/4.0,
             -(1-x)*(1+y)*xy/4.0,
             -(1-x)*(1+x)*(1-y)*y/2.0,
              (1+x)*(1-y)*(1+y)*x/2.0,
              (1-x)*(1+x)*(1+y)*y/2.0,
             -(1-x)*(1-y)*(1+y)*x/2.0,
              (1-x)*(1+x)*(1-y)*(1+y)};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto xy = x*y;

        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            { (1-2*x)*(1-y)*y/4.0,  (1-x)*(1-2*y)*x/4.0,
             -(1+2*x)*(1-y)*y/4.0, -(1+x)*(1-2*y)*x/4.0,
              (1+2*x)*(1+y)*y/4.0,  (1+x)*(1+2*y)*x/4.0,
             -(1-2*x)*(1+y)*y/4.0, -(1-x)*(1+2*y)*x/4.0,

              (1-y)*xy,                -(1-x)*(1+x)*(1-2*y)/2.0,
              (1+2*x)*(1-y)*(1+y)/2.0, -(1+x)*xy,
             -(1+y)*xy,                 (1-x)*(1+x)*(1+2*y)/2.0,
             -(1-2*x)*(1-y)*(1+y)/2.0,  (1-x)*xy,
             -2*(1-y)*(1+y)*x,         -2*(1-x)*(1+x)*y};
    }

    template<typename ScalarType>
    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto differentialMeasure(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    ) -> ScalarType
    {
        const ScalarType ax = aJacobian(0,1)*aJacobian(1,2)-aJacobian(0,2)*aJacobian(1,1);
        const ScalarType ay = aJacobian(0,2)*aJacobian(1,0)-aJacobian(0,0)*aJacobian(1,2);
        const ScalarType az = aJacobian(0,0)*aJacobian(1,1)-aJacobian(0,1)*aJacobian(1,0);

        return sqrt(ax*ax+ay*ay+az*az);
    }

    template<typename ScalarType>
    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto differentialVector(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    ) -> Plato::Array<mNumSpatialDims+1, ScalarType>
    {
        return Plato::Array<mNumSpatialDims+1, ScalarType>
            {aJacobian(0,1)*aJacobian(1,2)-aJacobian(0,2)*aJacobian(1,1),
             aJacobian(0,2)*aJacobian(1,0)-aJacobian(0,0)*aJacobian(1,2),
             aJacobian(0,0)*aJacobian(1,1)-aJacobian(0,1)*aJacobian(1,0)};
    }
};

} // end namespace Plato
