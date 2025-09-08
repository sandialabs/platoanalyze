#pragma once

#include "PlatoMathTypes.hpp"
#include "Tet4.hpp"
#include "Tri6.hpp"

namespace Plato
{

/******************************************************************************/
/*! Tet10 Element
 */
/******************************************************************************/
class Tet10
{
   public:
    using Face = Plato::Tri6;
    using C1 = Plato::Tet4;

    static constexpr Plato::OrdinalType mNumSpatialDims = 3;
    static constexpr Plato::OrdinalType mNumNodesPerCell = 10;
    static constexpr Plato::OrdinalType mNumNodesPerFace = 6;
    static constexpr Plato::OrdinalType mNumGaussPoints = 16;

    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = mNumSpatialDims - 1;

    static constexpr Plato::Array<mNumGaussPoints> getCubWeights()
    {
        constexpr double w1 = 8.395632350020469e-03;
        constexpr double w2 = 1.109034477221540e-02;
        return Plato::Array<mNumGaussPoints>({w1, w1, w1, w1, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2, w2});
    }

    static constexpr Plato::Matrix<mNumGaussPoints, mNumSpatialDims> getCubPoints()
    {
        constexpr double u1 = 0.7716429020672371;
        constexpr double u2 = 0.07611903264425430;
        constexpr double u3 = 0.4042339134672644;
        constexpr double u4 = 0.1197005277978019;
        constexpr double u5 = 0.07183164526766925;
        return Plato::Matrix<mNumGaussPoints, mNumSpatialDims>(
            {u1, u2, u2, u2, u1, u2, u2, u2, u1, u2, u2, u2, u3, u5, u4, u3, u4, u5, u4, u5, u3, u4, u3, u5,
             u5, u4, u3, u5, u3, u4, u3, u5, u3, u5, u3, u3, u3, u3, u5, u4, u3, u3, u3, u4, u3, u3, u3, u4});
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static auto basisValues(
        const Plato::Array<mNumSpatialDims>& aCubPoint) -> Plato::Array<mNumNodesPerCell>
    {
        const auto x = aCubPoint(0);
        const auto y = aCubPoint(1);
        const auto z = aCubPoint(2);

        const auto tCon = (x + y + z - 1.0);

        return Plato::Array<mNumNodesPerCell>{tCon * (2.0 * x + 2.0 * y + 2.0 * z - 1),
                                              x * (2.0 * x - 1.0),
                                              y * (2.0 * y - 1.0),
                                              z * (2.0 * z - 1.0),
                                              -4.0 * x * tCon,
                                              4.0 * x * y,
                                              -4.0 * y * tCon,
                                              -4.0 * z * tCon,
                                              4.0 * x * z,
                                              4.0 * y * z};
    }

    [[nodiscard]] constexpr KOKKOS_INLINE_FUNCTION static auto basisGrads(
        const Plato::Array<mNumSpatialDims>& aCubPoint) -> Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>
    {
        const auto x = aCubPoint(0);
        const auto y = aCubPoint(1);
        const auto z = aCubPoint(2);

        return Plato::Matrix<mNumNodesPerCell, mNumSpatialDims>{(x + y + z) * 4.0 - 3.0,
                                                                (x + y + z) * 4.0 - 3.0,
                                                                (x + y + z) * 4.0 - 3.0,
                                                                x * 4.0 - 1.0,
                                                                0.0,
                                                                0.0,
                                                                0.0,
                                                                y * 4.0 - 1.0,
                                                                0.0,
                                                                0.0,
                                                                0.0,
                                                                z * 4.0 - 1.0,
                                                                -(x * 2.0 + y + z - 1) * 4.0,
                                                                -x * 4.0,
                                                                -x * 4.0,
                                                                y * 4.0,
                                                                x * 4.0,
                                                                0.0,
                                                                -y * 4.0,
                                                                -(x + 2.0 * y + z - 1) * 4.0,
                                                                -y * 4.0,
                                                                -z * 4.0,
                                                                -z * 4.0,
                                                                -(x + y + z * 2.0 - 1) * 4.,
                                                                z * 4.0,
                                                                0.0,
                                                                x * 4.,
                                                                0.0,
                                                                z * 4.0,
                                                                y * 4.};
    }
};

}  // end namespace Plato
