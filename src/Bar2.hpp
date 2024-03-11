#pragma once

#include "PlatoMathTypes.hpp"

namespace Plato {

/******************************************************************************/
/*! Bar2 Element
*/
/******************************************************************************/
class Bar2
{
  public:

    static constexpr Plato::OrdinalType mNumSpatialDims  = 1;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 2;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 1;
    static constexpr Plato::OrdinalType mNumGaussPoints  = 2;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims-1;

    [[nodiscard]] static constexpr Plato::Array<mNumGaussPoints>
    getCubWeights()
    {
        return Plato::Array<mNumGaussPoints>{
            Plato::Scalar(1.0), Plato::Scalar(1.0)
        };
    }

    [[nodiscard]] static constexpr Plato::Matrix<mNumGaussPoints,mNumSpatialDims>
    getCubPoints()
    {
        constexpr Plato::Scalar sqt = 0.57735026918962584208117050366127; // sqrt(1.0/3.0)
        return Plato::Matrix<mNumGaussPoints,mNumSpatialDims>{ -sqt,  sqt };
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisValues( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Array<mNumNodesPerCell>
    {
        const auto x=aCubPoint(0);

        return Plato::Array<mNumNodesPerCell>
            {(1-x)/2.0,
             (1+x)/2.0};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static 
    auto basisGrads( const Plato::Array<mNumSpatialDims>& aCubPoint ) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
            {Plato::Scalar(-1)/2.0,
             Plato::Scalar(1)/2.0};
    }

    template<typename ScalarType>
    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static
    auto differentialMeasure(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    ) -> ScalarType
    {
        const ScalarType ax = aJacobian(0,0);
        const ScalarType ay = aJacobian(0,1);

        return sqrt(ax*ax+ay*ay);
    }

    template<typename ScalarType>
    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static
    auto differentialVector(
        const Plato::Matrix<mNumSpatialDims, mNumSpatialDims+1, ScalarType> & aJacobian
    ) -> Plato::Array<mNumSpatialDims+1, ScalarType>
    {
        return Plato::Array<mNumSpatialDims+1, ScalarType>
            {aJacobian(0,0),
             aJacobian(0,1)};
    }
};

} // end namespace Plato
