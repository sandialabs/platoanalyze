#ifndef PLATO_COMPOSABLEFUNCTIONOBJECTS_MATERIAL_STRESSTRANSLATOR_H
#define PLATO_COMPOSABLEFUNCTIONOBJECTS_MATERIAL_STRESSTRANSLATOR_H

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"

namespace plato::composable_function_objects::material
{
/// @brief class to translate between different stress measures in finite deformation mechanics
template <typename ValueType, Plato::OrdinalType VoigtTerms>
class StressTranslator
{
   private:
    static constexpr Plato::OrdinalType mTensorDim{3};

   public:
    /// @brief translate the input first Piola-Kirchhoff stress @param aP to Cauchy stress using deformation gradient
    /// @param aF. The cauchy stress is stored in Voigt notation.
    template <typename StrainType>
    KOKKOS_INLINE_FUNCTION Plato::Array<VoigtTerms, ValueType> cauchyStressFromFirstPiolaKirchhoffStress(
        const Plato::Matrix<mTensorDim, mTensorDim, ValueType>& aP,
        const Plato::Matrix<mTensorDim, mTensorDim, StrainType>& aF) const;
};

namespace detail
{
KOKKOS_INLINE_FUNCTION Plato::Matrix<3, 3, Plato::Scalar> full_cauchy_stress_from_first_piola_kirchhoff_stress(
    const Plato::Matrix<3, 3, Plato::Scalar>& aP, const Plato::Matrix<3, 3, Plato::Scalar>& aF);
}  // namespace detail

/// @brief base implementation just returns an array of 0s since this shouldn't be used with FAD types
template <typename ValueType, Plato::OrdinalType VoigtTerms>
template <typename StrainType>
Plato::Array<VoigtTerms, ValueType> StressTranslator<ValueType, VoigtTerms>::cauchyStressFromFirstPiolaKirchhoffStress(
    const Plato::Matrix<mTensorDim, mTensorDim, ValueType>& aP,
    const Plato::Matrix<mTensorDim, mTensorDim, StrainType>& aF) const
{
    return Plato::Array<VoigtTerms, ValueType>(0.0);
}

/// @brief specialized implementation for scalar types in 2D
template <>
template <>
Plato::Array<3, Plato::Scalar> StressTranslator<Plato::Scalar, 3>::cauchyStressFromFirstPiolaKirchhoffStress(
    const Plato::Matrix<mTensorDim, mTensorDim, Plato::Scalar>& aP,
    const Plato::Matrix<mTensorDim, mTensorDim, Plato::Scalar>& aF) const
{
    const auto tCauchyStress = detail::full_cauchy_stress_from_first_piola_kirchhoff_stress(aP, aF);
    return Plato::Array<3, Plato::Scalar>{tCauchyStress(0, 0), tCauchyStress(1, 1), tCauchyStress(0, 1)};
}

/// @brief specialized implementation for scalar types in 3D
template <>
template <>
Plato::Array<6, Plato::Scalar> StressTranslator<Plato::Scalar, 6>::cauchyStressFromFirstPiolaKirchhoffStress(
    const Plato::Matrix<mTensorDim, mTensorDim, Plato::Scalar>& aP,
    const Plato::Matrix<mTensorDim, mTensorDim, Plato::Scalar>& aF) const
{
    const auto tCauchyStress = detail::full_cauchy_stress_from_first_piola_kirchhoff_stress(aP, aF);
    return Plato::Array<6, Plato::Scalar>{tCauchyStress(0, 0), tCauchyStress(1, 1), tCauchyStress(2, 2),
                                          tCauchyStress(1, 2), tCauchyStress(0, 2), tCauchyStress(0, 1)};
}

namespace detail
{
Plato::Matrix<3, 3, Plato::Scalar> full_cauchy_stress_from_first_piola_kirchhoff_stress(
    const Plato::Matrix<3, 3, Plato::Scalar>& aP, const Plato::Matrix<3, 3, Plato::Scalar>& aF)
{
    const auto tFT = Plato::transpose(aF);
    const auto tJ = Plato::determinant(aF);
    return Plato::times(1.0 / tJ, Plato::times(aP, tFT));
}
}  // namespace detail
}  // namespace plato::composable_function_objects::material

#endif
