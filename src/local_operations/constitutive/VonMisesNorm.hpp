#ifndef PLATO_LOCAL_OPERATIONS_CONSTITUTIVE_VONMISESNORM
#define PLATO_LOCAL_OPERATIONS_CONSTITUTIVE_VONMISESNORM

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"
#include "local_operations/constitutive/VonMisesYieldFunction.hpp"

namespace Plato
{
/// @brief class for computing the von mises stress measure
struct VonMisesNorm
{
    template <Plato::OrdinalType VoigtLength, typename ResultScalarType>
    KOKKOS_INLINE_FUNCTION ResultScalarType
    operator()(Plato::Array<VoigtLength, ResultScalarType> aElementAveragedVoigtStress) const
    {
        constexpr Plato::OrdinalType tSpatialDims =
            (VoigtLength == 6) ? 3 : ((VoigtLength == 3) ? 2 : (((VoigtLength == 1) ? 1 : 0)));
        const Plato::VonMisesYieldFunction<tSpatialDims, VoigtLength> tComputeVonMises;

        ResultScalarType tVonMisesStress{0.0};
        tComputeVonMises(aElementAveragedVoigtStress, tVonMisesStress);
        return tVonMisesStress;
    }
};
}  // namespace Plato

#endif
