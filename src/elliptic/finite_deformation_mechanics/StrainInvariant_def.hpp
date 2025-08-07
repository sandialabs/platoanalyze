#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_STRAININVARIANT_DEF_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_STRAININVARIANT_DEF_H

#include <Teuchos_ParameterList.hpp>
#include <string>

#include "FadTypes.hpp"
#include "GradientMatrix.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTypes.hpp"
#include "SpatialModel.hpp"
#include "composable_function_objects/kinematics/DeformationGradient.hpp"
#include "composable_function_objects/material/NeoHookeanModel.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/finite_deformation_mechanics/StrainInvariant_decl.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
template <typename EvaluationType, typename IndicatorFunctionType>
StrainInvariant<EvaluationType, IndicatorFunctionType>::StrainInvariant(const Plato::SpatialDomain& aSpatialDomain,
                                                                        Plato::DataMap& aDataMap,
                                                                        Teuchos::ParameterList& aProblemParams,
                                                                        Teuchos::ParameterList& aPenaltyParams,
                                                                        const std::string& aFunctionName)
    : FunctionBaseType(aSpatialDomain, aDataMap, aProblemParams, aFunctionName),
      mIndicatorFunction(aPenaltyParams),
      mApplyWeighting(mIndicatorFunction),
      mNeoHookeanParameters{composable_function_objects::material::get_neo_hookean_parameters(
          aProblemParams.sublist("Material Models").sublist(aSpatialDomain.getMaterialName()))}
{
}

template <typename EvaluationType, typename IndicatorFunctionType>
void StrainInvariant<EvaluationType, IndicatorFunctionType>::evaluate_conditional(
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
    material::NeoHookeanModel tMaterialModel{mNeoHookeanParameters};
    auto& tComputeScaleFactorFromControl = mApplyWeighting;

    const auto tCubPoints = ElementType::getCubPoints();
    const auto tCubWeights = ElementType::getCubWeights();

    const auto tNumPoints = tCubWeights.size();
    const auto tNumCells = mSpatialDomain.numCells();
    Kokkos::parallel_for(
        "compute average strain invariant over each element",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType tCellOrdinal, const Plato::OrdinalType tGpOrdinal) {
            const auto tCubPoint = tCubPoints(tGpOrdinal);

            ConfigScalarType tVolume(0.0);
            Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tShapeGradients;
            tComputeGradient(tCellOrdinal, tCubPoint, aConfig, tShapeGradients, tVolume);
            tVolume *= tCubWeights(tGpOrdinal);

            Plato::Matrix<mTensorDim, mTensorDim, StrainScalarType> tDeformationGradient(0.0);
            tComputeDeformationGradient(tCellOrdinal, tDeformationGradient, aState, tShapeGradients);

            // TODO: generalize invariants
            ResultScalarType tInvariant = std::pow(Plato::norm(tDeformationGradient), 2.0);  // I1 = F:F

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ResultScalarType tScaleFactor{1.0};
            tComputeScaleFactorFromControl(tCellOrdinal, aControl, tBasisValues, tScaleFactor);

            Kokkos::atomic_add(&aResult(tCellOrdinal),
                               tScaleFactor * tInvariant / tNumPoints);  // take average over element
        });
}
};  // namespace plato::elliptic::finite_deformation_mechanics
#endif
