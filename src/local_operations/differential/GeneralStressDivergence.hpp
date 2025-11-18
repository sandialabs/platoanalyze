#ifndef PLATO_COMPOSABLEFUNCTIONOBJECTS_SHAPEFUNCTIONOPERATIONS_GENERALSTRESSDIVERGENCE_H
#define PLATO_COMPOSABLEFUNCTIONOBJECTS_SHAPEFUNCTIONOPERATIONS_GENERALSTRESSDIVERGENCE_H

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace plato::composable_function_objects::shape_function_operations
{
/// @brief Stress Divergence functor. Given the stress tensor, apply the divergence operator to the stress tensor.
/// @tparam ElementType Base element type
/// @tparam NumDofsPerNode number of degrees of freedom per node
/// @tparam DofOffset offset apply to degree of freedom indexing
template <typename ElementType,
          Plato::OrdinalType NumDofsPerNode = ElementType::mNumSpatialDims,
          Plato::OrdinalType DofOffset = 0>
class GeneralStressDivergence : public ElementType
{
   private:
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

   public:
    /// @brief default constructor
    GeneralStressDivergence();

    /// @brief Overload to apply stress divergence operator to stress tensor stored as an array with Voigt notation
    /// Sacado FAD types are deduced from arguments
    /// @tparam ForcingScalarType Kokkos::View POD type
    /// @tparam StressScalarType Kokkos::View POD type
    /// @tparam GradientScalarType Kokkos::View POD type
    /// @tparam VolumeScalarType Kokkos::View POD type
    /// @param aCellOrdinal cell index
    /// @param aOutput view that stores stress divergence values
    /// @param aStress stress tensor
    /// @param aGradient shape function gradients
    /// @param aCellVolume cell volume
    /// @param aScale multiplier(default = 1.0)
    template <typename ForcingScalarType,
              typename StressScalarType,
              typename GradientScalarType,
              typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(
        const Plato::OrdinalType& aCellOrdinal,
        const Plato::ScalarMultiVectorT<ForcingScalarType>& aOutput,
        const Plato::Array<mNumVoigtTerms, StressScalarType>& aStress,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType>& aGradient,
        const VolumeScalarType& aCellVolume,
        const Plato::Scalar aScale = 1.0) const;

    /// @brief Overload to apply stress divergence operator to stress tensor stored as a matrix with all components
    template <typename ForcingScalarType,
              typename StressScalarType,
              typename GradientScalarType,
              typename VolumeScalarType,
              Plato::OrdinalType StressDims>
    KOKKOS_INLINE_FUNCTION void operator()(
        const Plato::OrdinalType& aCellOrdinal,
        const Plato::ScalarMultiVectorT<ForcingScalarType>& aOutput,
        const Plato::Matrix<StressDims, StressDims, StressScalarType>& aStress,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType>& aGradient,
        const VolumeScalarType& aCellVolume,
        const Plato::Scalar aScale = 1.0) const;

    /// @brief Overload to apply stress divergence operator to stress tensor stored in a 3D view with Voigt notation
    template <typename ForcingScalarType,
              typename StressScalarType,
              typename GradientScalarType,
              typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(const Plato::OrdinalType& aCellOrdinal,
                                           const Plato::OrdinalType& aGpOrdinal,
                                           const Plato::ScalarMultiVectorT<ForcingScalarType>& aOutput,
                                           const Plato::ScalarArray3DT<StressScalarType>& aStress,
                                           const Plato::ScalarArray4DT<GradientScalarType>& aGradient,
                                           const Plato::ScalarMultiVectorT<VolumeScalarType>& aCellVolume,
                                           const Plato::Scalar aScale = 1.0) const;

   private:
    /// @brief general function to apply stress divergence operator to stress stored in different ways
    /// @tparam StressDivergenceFunc callable for the stress contribution.
    ///         Must have the following signature:
    ///         ForcingScalarType(Plato::OrdinalType aDimIndexI, Plato::OrdinalType aDimIndexJ, Plato::OrdinalType
    ///         aNodeIndex)
    template <typename ForcingScalarType, typename StressDivergenceFunc>
    KOKKOS_INLINE_FUNCTION void applyStressDivergence(const Plato::OrdinalType& aCellOrdinal,
                                                      const Plato::ScalarMultiVectorT<ForcingScalarType>& aOutput,
                                                      const StressDivergenceFunc& aComputeDivergence) const;

   private:
    // mapping from matrix to Voigt notation
    // 2-D Example: mVoigt[0][0] = 0, mVoigt[0][1] = 2, mVoigt[1][0] = 2, mVoigt[1][1] = 1,
    Plato::Matrix<mNumSpatialDims, mNumSpatialDims, Plato::OrdinalType>
        mVoigt;  // matrix with indices to stress tensor entries in Voigt notation
};

namespace detail
{
Plato::Matrix<2, 2, Plato::OrdinalType> voigt_map_2d();

Plato::Matrix<3, 3, Plato::OrdinalType> voigt_map_3d();
}  // namespace detail

template <typename ElementType, Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType DofOffset>
GeneralStressDivergence<ElementType, NumDofsPerNode, DofOffset>::GeneralStressDivergence()
{
    if constexpr (mNumSpatialDims == 2)
    {
        mVoigt = detail::voigt_map_2d();
    }
    else if constexpr (mNumSpatialDims == 3)
    {
        mVoigt = detail::voigt_map_3d();
    }
}

template <typename ElementType, Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType DofOffset>
template <typename ForcingScalarType, typename StressScalarType, typename GradientScalarType, typename VolumeScalarType>
void GeneralStressDivergence<ElementType, NumDofsPerNode, DofOffset>::operator()(
    const Plato::OrdinalType& aCellOrdinal,
    const Plato::ScalarMultiVectorT<ForcingScalarType>& aOutput,
    const Plato::Array<mNumVoigtTerms, StressScalarType>& aStress,
    const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType>& aGradient,
    const VolumeScalarType& aCellVolume,
    const Plato::Scalar aScale) const
{
    auto tComputeStressDivergence = [aScale, &aCellVolume, &aStress, &aGradient, &tVoigt = mVoigt](
                                        Plato::OrdinalType aDimIndexI, Plato::OrdinalType aDimIndexJ,
                                        Plato::OrdinalType aNodeIndex) -> ForcingScalarType
    { return aScale * aCellVolume * aStress(tVoigt(aDimIndexI, aDimIndexJ)) * aGradient(aNodeIndex, aDimIndexJ); };
    applyStressDivergence(aCellOrdinal, aOutput, tComputeStressDivergence);
}

template <typename ElementType, Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType DofOffset>
template <typename ForcingScalarType,
          typename StressScalarType,
          typename GradientScalarType,
          typename VolumeScalarType,
          Plato::OrdinalType StressDims>
void GeneralStressDivergence<ElementType, NumDofsPerNode, DofOffset>::operator()(
    const Plato::OrdinalType& aCellOrdinal,
    const Plato::ScalarMultiVectorT<ForcingScalarType>& aOutput,
    const Plato::Matrix<StressDims, StressDims, StressScalarType>& aStress,
    const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType>& aGradient,
    const VolumeScalarType& aCellVolume,
    const Plato::Scalar aScale) const
{
    auto tComputeStressDivergence = [aScale, &aCellVolume, &aStress, &aGradient](
                                        Plato::OrdinalType aDimIndexI, Plato::OrdinalType aDimIndexJ,
                                        Plato::OrdinalType aNodeIndex) -> ForcingScalarType
    { return aScale * aCellVolume * aStress(aDimIndexI, aDimIndexJ) * aGradient(aNodeIndex, aDimIndexJ); };
    applyStressDivergence(aCellOrdinal, aOutput, tComputeStressDivergence);
}

template <typename ElementType, Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType DofOffset>
template <typename ForcingScalarType, typename StressScalarType, typename GradientScalarType, typename VolumeScalarType>
void GeneralStressDivergence<ElementType, NumDofsPerNode, DofOffset>::operator()(
    const Plato::OrdinalType& aCellOrdinal,
    const Plato::OrdinalType& aGpOrdinal,
    const Plato::ScalarMultiVectorT<ForcingScalarType>& aOutput,
    const Plato::ScalarArray3DT<StressScalarType>& aStress,
    const Plato::ScalarArray4DT<GradientScalarType>& aGradient,
    const Plato::ScalarMultiVectorT<VolumeScalarType>& aCellVolume,
    const Plato::Scalar aScale) const
{
    auto tComputeStressDivergence = [aCellOrdinal, aGpOrdinal, aScale, &aCellVolume, &aStress, &aGradient,
                                     &tVoigt = mVoigt](Plato::OrdinalType aDimIndexI, Plato::OrdinalType aDimIndexJ,
                                                       Plato::OrdinalType aNodeIndex) -> ForcingScalarType
    {
        return aScale * aCellVolume(aCellOrdinal, aGpOrdinal) *
               aStress(aCellOrdinal, aGpOrdinal, tVoigt(aDimIndexI, aDimIndexJ)) *
               aGradient(aCellOrdinal, aGpOrdinal, aNodeIndex, aDimIndexJ);
    };
    applyStressDivergence(aCellOrdinal, aOutput, tComputeStressDivergence);
}

template <typename ElementType, Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType DofOffset>
template <typename ForcingScalarType, typename StressDivergenceFunc>
void GeneralStressDivergence<ElementType, NumDofsPerNode, DofOffset>::applyStressDivergence(
    const Plato::OrdinalType& aCellOrdinal,
    const Plato::ScalarMultiVectorT<ForcingScalarType>& aOutput,
    const StressDivergenceFunc& aComputeDivergence) const
{
    for (Plato::OrdinalType tDimIndexI = 0; tDimIndexI < mNumSpatialDims; tDimIndexI++)
    {
        for (Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + tDimIndexI + DofOffset;
            ForcingScalarType tLocalDivergence{0.0};
            for (Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < mNumSpatialDims; tDimIndexJ++)
            {
                tLocalDivergence += aComputeDivergence(tDimIndexI, tDimIndexJ, tNodeIndex);
            }
            Kokkos::atomic_add(&aOutput(aCellOrdinal, tLocalOrdinal), tLocalDivergence);
        }
    }
}
}  // namespace plato::composable_function_objects::shape_function_operations

#endif
