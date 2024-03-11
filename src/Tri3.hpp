#pragma once

#include "Bar2.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Tri3 Element
*/
/******************************************************************************/
class Tri3
{
  public:
    using Face = Plato::Bar2;

    static constexpr Plato::OrdinalType mNumSpatialDims  = 2;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 3;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 2;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 1;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights() { return Plato::Array<mNumGaussPoints>({Plato::Scalar(1)/2}); }

    static constexpr Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>({
            Plato::Scalar(1)/3, Plato::Scalar(1)/3
        });
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Array<mNumNodesPerCell>
    {
        const auto x=aCubPoint(0);
        const auto y=aCubPoint(1);

        return Plato::Array<mNumNodesPerCell>{1-x-y, x, y};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            {-1, -1, 
              1, 0, 
              0, 1};
    }

    template<typename ScalarType>
    constexpr KOKKOS_INLINE_FUNCTION static 
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
    constexpr KOKKOS_INLINE_FUNCTION static 
    auto differentialVector(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    ) -> Plato::Array<mNumSpatialDims+1, ScalarType>
    {
        return Plato::Array<mNumSpatialDims+1, ScalarType>
        { aJacobian(0,1)*aJacobian(1,2)-aJacobian(0,2)*aJacobian(1,1),
          aJacobian(0,2)*aJacobian(1,0)-aJacobian(0,0)*aJacobian(1,2),
          aJacobian(0,0)*aJacobian(1,1)-aJacobian(0,1)*aJacobian(1,0)};
    }
};

} // end namespace Plato
