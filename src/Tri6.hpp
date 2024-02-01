#pragma once

#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tri6 Element

    See Klaus-Jurgen Bathe, "Finite Element Proceedures", 2006, pg 374.
*/
/******************************************************************************/
class Tri6
{
  public:
    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 6;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 3;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 3;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    [[nodiscard]] static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights() { return Plato::Array<mNumGaussPoints>({Plato::Scalar(1.0)/6.0, Plato::Scalar(1.0)/6.0, Plato::Scalar(1.0)/6.0}); }

    [[nodiscard]] static constexpr Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>{
            Plato::Scalar(2.0)/3, Plato::Scalar(1.0)/6, 
            Plato::Scalar(1.0)/6, Plato::Scalar(2.0)/3, 
            Plato::Scalar(1.0)/6, Plato::Scalar(1.0)/6 
        };
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static Plato::Array<mNumNodesPerCell>
    basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);
        const auto x2=x*x;
        const auto y2=y*y;
        const auto xy=x*y;

        return Plato::Array<mNumNodesPerCell>
            {1-3*(x+y)+2*(x2+y2)+4*xy,
             2*x2-x,
             2*y2-y,
             4*(x-xy-x2),
             4*xy,
             4*(y-xy-y2)};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint )
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);

        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            { 4*(x+y)-3,  4*(x+y)-3,
              4*x-1    ,  0,
              0        ,  4*y-1,
              4-8*x-4*y, -4*x,
              4*y      ,  4*x,
             -4*y      ,  4-4*x-8*y};
    }

    template<typename ScalarType>
    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    ScalarType differentialMeasure(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        const ScalarType ax = aJacobian(0,1)*aJacobian(1,2)-aJacobian(0,2)*aJacobian(1,1);
        const ScalarType ay = aJacobian(0,2)*aJacobian(1,0)-aJacobian(0,0)*aJacobian(1,2);
        const ScalarType az = aJacobian(0,0)*aJacobian(1,1)-aJacobian(0,1)*aJacobian(1,0);

        return sqrt(ax*ax+ay*ay+az*az);
    }

    template<typename ScalarType>
    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    Plato::Array<mNumSpatialDims+1, ScalarType>
    differentialVector(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    )
    {
        return Plato::Array<mNumSpatialDims+1, ScalarType>
            {aJacobian(0,1)*aJacobian(1,2)-aJacobian(0,2)*aJacobian(1,1),
             aJacobian(0,2)*aJacobian(1,0)-aJacobian(0,0)*aJacobian(1,2),
             aJacobian(0,0)*aJacobian(1,1)-aJacobian(0,1)*aJacobian(1,0)};
    }
};

} // end namespace Plato
