#ifndef PLATO_COMPOSABLEFUNCTIONOBJECTS_MATERIAL_NEOHOOKEANMODEL_H
#define PLATO_COMPOSABLEFUNCTIONOBJECTS_MATERIAL_NEOHOOKEANMODEL_H

#include <Kokkos_Macros.hpp>
#include <Teuchos_ParameterList.hpp>
#include <cmath>

#include "PlatoMathTypes.hpp"
#include "PlatoTypes.hpp"

namespace plato::composable_function_objects::material
{
/// @brief struct for storing Neo Hookean model parameters
struct NeoHookeanParameters
{
    Plato::Scalar mBulkModulus;
    Plato::Scalar mShearModulus;
};

NeoHookeanParameters get_neo_hookean_parameters(const Teuchos::ParameterList& aMaterialParamList);

/// @brief Function object to model hyperelastic response from a free energy with Neo-Hookean form:
/// W(C) = 1/2 G (J^(-2/3)tr(C) - 3) + 1/2 K (1/2 J^2 - 1/2 - ln(J)).
class NeoHookeanModel
{
   private:
    static constexpr Plato::OrdinalType mTensorDim{3};

   public:
    explicit NeoHookeanModel(const NeoHookeanParameters& aParameters);

    /// @brief compute the strain energy @param aEnergy from deformation gradient @param aDeformationGradient.
    /// @param aEnergy is a return parameter so that the correct FAD type @a EnergyType can be
    /// deduced.
    template <typename StrainType, typename EnergyType>
    KOKKOS_INLINE_FUNCTION void energy(const Plato::Matrix<mTensorDim, mTensorDim, StrainType>& aDeformationGradient,
                                       EnergyType& aEnergy) const;

    /// @brief compute the first Piola-Kirchhoff stress @param aStress from deformation gradient @param
    /// aDeformationGradient.
    /// @param aStress is a return parameter so that the correct FAD type @a StressType can be
    /// deduced.
    template <typename StrainType, typename StressType>
    KOKKOS_INLINE_FUNCTION void stress(const Plato::Matrix<mTensorDim, mTensorDim, StrainType>& aDeformationGradient,
                                       Plato::Matrix<mTensorDim, mTensorDim, StressType>& aStress) const;

   private:
    NeoHookeanParameters mParameters;
};

template <typename StrainType, typename EnergyType>
void NeoHookeanModel::energy(const Plato::Matrix<mTensorDim, mTensorDim, StrainType>& aDeformationGradient,
                             EnergyType& aEnergy) const
{
    const StrainType tJ = Plato::determinant(aDeformationGradient);
    const StrainType tJ23 = std::pow(tJ, -2.0 / 3.0);

    const StrainType tFNorm = Plato::norm(aDeformationGradient);
    const StrainType tI1 = tFNorm * tFNorm;  // I1 = F:F

    aEnergy += 0.5 * mParameters.mShearModulus * (tJ23 * tI1 - 3.0);
    aEnergy += 0.5 * mParameters.mBulkModulus * (0.5 * tJ * tJ - 0.5 - std::log(tJ));
}

template <typename StrainType, typename StressType>
void NeoHookeanModel::stress(const Plato::Matrix<mTensorDim, mTensorDim, StrainType>& aDeformationGradient,
                             Plato::Matrix<mTensorDim, mTensorDim, StressType>& aStress) const
{
    const StrainType tJ = Plato::determinant(aDeformationGradient);
    const StrainType tJ23 = std::pow(tJ, -2.0 / 3.0);

    const StrainType tFNorm = Plato::norm(aDeformationGradient);
    const StrainType tI1 = tFNorm * tFNorm;  // I1 = F:F

    const Plato::Matrix<mTensorDim, mTensorDim, StrainType> tFinv = Plato::invert(aDeformationGradient);
    const Plato::Matrix<mTensorDim, mTensorDim, StrainType> tFinvT = Plato::transpose(tFinv);

    const StrainType c0 = -tI1 / 3.0;
    const StrainType c1 = mParameters.mShearModulus * tJ23;
    const StrainType c2 = 0.5 * mParameters.mBulkModulus * (tJ * tJ - 1.0);
    const auto tPIso = Plato::times(c1, Plato::plus(aDeformationGradient, Plato::times(c0, tFinvT)));
    const auto tPVol = Plato::times(c2, tFinvT);

    for (Plato::OrdinalType i = 0; i < mTensorDim; i++)
    {
        for (Plato::OrdinalType j = 0; j < mTensorDim; j++)
        {
            aStress(i, j) += tPIso(i, j) + tPVol(i, j);
        }
    }
}
}  // namespace plato::composable_function_objects::material

#endif
