#ifndef PLATO_COMPOSABLEFUNCTIONOBJECTS_KINEMATICS_DEFORMATIONGRADIENT_H
#define PLATO_COMPOSABLEFUNCTIONOBJECTS_KINEMATICS_DEFORMATIONGRADIENT_H

#include <Kokkos_Macros.hpp>

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace plato::composable_function_objects::kinematics
{
/// @brief Function object to compute the deformation gradient from displacement field.
/// this is stored as a Plato::Matrix with size 3x3
template <typename ElementType>
class DeformationGradient : public ElementType
{
   private:
    static constexpr Plato::OrdinalType mTensorDim{3};
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

   public:
    /// @brief callable that computes the deformation gradient @param aDeformationGradient from displacement field
    /// stored in @param aState and shape function gradients in @param aGradient.
    /// @param aDeformationGradient is a return parameter so that the correct FAD type @a StrainScalarType can be
    /// deduced.
    template <typename StrainScalarType, typename DispScalarType, typename GradientScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::Matrix<mTensorDim, mTensorDim, StrainScalarType>& aDeformationGradient,
        const Plato::ScalarMultiVectorT<DispScalarType>& aState,
        const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType>& aGradient) const;
};

template <typename ElementType>
template <typename StrainScalarType, typename DispScalarType, typename GradientScalarType>
void DeformationGradient<ElementType>::operator()(
    Plato::OrdinalType aCellOrdinal,
    Plato::Matrix<mTensorDim, mTensorDim, StrainScalarType>& aDeformationGradient,
    const Plato::ScalarMultiVectorT<DispScalarType>& aState,
    const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, GradientScalarType>& aGradient) const
{
    for (Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumDofsPerNode; tDofIndex++)
    {
        for (Plato::OrdinalType tDimIndex = 0; tDimIndex < mNumSpatialDims; tDimIndex++)
        {
            for (Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                const auto tStateOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex;
                aDeformationGradient(tDofIndex, tDimIndex) +=
                    aState(aCellOrdinal, tStateOrdinal) * aGradient(tNodeIndex, tDimIndex);
            }
        }
    }

    for (Plato::OrdinalType tDiagIndex = 0; tDiagIndex < mTensorDim; tDiagIndex++)
    {
        aDeformationGradient(tDiagIndex, tDiagIndex) += 1.0;
    }
}
}  // namespace plato::composable_function_objects::kinematics

#endif
