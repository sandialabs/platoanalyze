#ifndef PLATO_LOCAL_OPERATIONS_CONSTITUTIVE_VOIGTTENSORL2NORM
#define PLATO_LOCAL_OPERATIONS_CONSTITUTIVE_VOIGTTENSORL2NORM

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"

namespace Plato
{
/// @brief struct for computing the l2 norm of a tensor in voigt notation
struct VoigtTensorL2Norm
{
    template <Plato::OrdinalType VoigtLength, typename ResultScalarType>
    KOKKOS_INLINE_FUNCTION ResultScalarType
    operator()(Plato::Array<VoigtLength, ResultScalarType> aElementAveragedVoigtStress) const
    {
        ResultScalarType tL2Norm{0.0};
        for (Plato::OrdinalType iVoigt = 0; iVoigt < VoigtLength; iVoigt++)
        {
            tL2Norm += aElementAveragedVoigtStress(iVoigt) * aElementAveragedVoigtStress(iVoigt);
        }
        return pow(tL2Norm, 1.0 / 2.0);
    }
};

}  // namespace Plato

#endif
