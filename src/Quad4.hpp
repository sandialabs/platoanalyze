#pragma once

#include "Bar2.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Quad4 Element
*/
/******************************************************************************/
class Quad4
{
  public:
    using Face = Plato::Bar2;
    using C1 = Plato::Quad4;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 4;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 2;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 4;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    [[nodiscard]] static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        return Plato::Array<mNumGaussPoints>{
            Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0), Plato::Scalar(1.0)
        };
    }

    [[nodiscard]] static constexpr Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        constexpr Plato::Scalar sqt = 0.57735026918962584208117050366127; // sqrt(1.0/3.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>{
            -sqt, -sqt,
             sqt, -sqt,
             sqt,  sqt,
            -sqt,  sqt
        };
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Array<mNumNodesPerCell>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);

        return Plato::Array<mNumNodesPerCell>
            {(1-x)*(1-y)/4.0,
             (1+x)*(1-y)/4.0,
             (1+x)*(1+y)/4.0,
             (1-x)*(1+y)/4.0};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);

        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            {-(1-y)/4.0, -(1-x)/4.0,
              (1-y)/4.0, -(1+x)/4.0,
              (1+y)/4.0,  (1+x)/4.0,
             -(1+y)/4.0,  (1-x)/4.0};
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
