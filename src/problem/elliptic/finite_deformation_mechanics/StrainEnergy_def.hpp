#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_STRAINENERGY_DEF_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_STRAINENERGY_DEF_H

#include <Teuchos_ParameterList.hpp>
#include <string>

#include "core_types/FadTypes.hpp"
#include "core_types/PlatoTypes.hpp"
#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "local_operations/differential/GradientMatrix.hpp"
#include "local_operations/kinematics/DeformationGradient.hpp"
#include "material/NeoHookeanModel.hpp"
#include "problem/elliptic/AbstractScalarFunction.hpp"
#include "problem/elliptic/finite_deformation_mechanics/StrainEnergy_decl.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
template <typename EvaluationType, typename IndicatorFunctionType>
StrainEnergy<EvaluationType, IndicatorFunctionType>::StrainEnergy(const Plato::SpatialDomain& aSpatialDomain,
                                                                  Plato::DataMap& aDataMap,
                                                                  Teuchos::ParameterList& aProblemParams,
                                                                  Teuchos::ParameterList& aPenaltyParams,
                                                                  const std::string& aFunctionName)
    : FunctionBaseType(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
      mIndicatorFunction(aPenaltyParams),
      mApplyWeighting(mIndicatorFunction)
{
    // get material parameters
    // Note: including composable_function_objects/material/NeoHookeanModel.hpp in the decl file causes weird errors
    // with std::pow and Fad types even though nothing changes in how they are used. Because of this,
    // NeoHookeanParameters can't be used as a member variable
    const auto tMaterialName = aSpatialDomain.getMaterialName();
    const auto tMaterialParameters = aProblemParams.sublist("Material Models").sublist(tMaterialName);
    const auto tNeoHookeanParameters =
        composable_function_objects::material::get_neo_hookean_parameters(tMaterialParameters);
    mKappa = tNeoHookeanParameters.mBulkModulus;
    mMu = tNeoHookeanParameters.mShearModulus;
}

template <typename EvaluationType, typename IndicatorFunctionType>
void StrainEnergy<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
{
    using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
    namespace kinematics = plato::composable_function_objects::kinematics;
    namespace material = plato::composable_function_objects::material;

    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    kinematics::DeformationGradient<ElementType> tComputeDeformationGradient;
    material::NeoHookeanModel tMaterialModel{material::NeoHookeanParameters{mKappa, mMu}};
    auto& tComputeScaleFactorFromControl = mApplyWeighting;

    const auto tCubPoints = ElementType::getCubPoints();
    const auto tCubWeights = ElementType::getCubWeights();

    const auto tNumPoints = tCubWeights.size();
    const auto tNumCells = mSpatialDomain.numCells();
    Kokkos::parallel_for(
        "compute strain energy", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType tCellOrdinal, const Plato::OrdinalType tGpOrdinal) {
            const auto tCubPoint = tCubPoints(tGpOrdinal);

            ConfigScalarType tVolume(0.0);
            Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tShapeGradients;
            tComputeGradient(tCellOrdinal, tCubPoint, aConfig, tShapeGradients, tVolume);
            tVolume *= tCubWeights(tGpOrdinal);

            Plato::Matrix<mTensorDim, mTensorDim, StrainScalarType> tDeformationGradient(0.0);
            tComputeDeformationGradient(tCellOrdinal, tDeformationGradient, aState, tShapeGradients);

            ResultScalarType tEnergy{0.0};
            tMaterialModel.energy(tDeformationGradient, tEnergy);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ResultScalarType tScaleFactor{1.0};
            tComputeScaleFactorFromControl(tCellOrdinal, aControl, tBasisValues, tScaleFactor);

            Kokkos::atomic_add(&aResult(tCellOrdinal), tScaleFactor * tEnergy * tVolume);
        });
}
};  // namespace plato::elliptic::finite_deformation_mechanics
#endif
