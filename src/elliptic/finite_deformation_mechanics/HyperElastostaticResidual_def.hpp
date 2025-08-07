#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_HYPERELASTOSTATICRESIDUAL_DEF_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_HYPERELASTOSTATICRESIDUAL_DEF_H

#include <Teuchos_ParameterList.hpp>
#include <memory>

#include "BodyLoads.hpp"
#include "FadTypes.hpp"
#include "GradientMatrix.hpp"
#include "NaturalBCs.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "Solutions.hpp"
#include "SpatialModel.hpp"
#include "ToMap.hpp"
#include "composable_function_objects/kinematics/DeformationGradient.hpp"
#include "composable_function_objects/material/NeoHookeanModel.hpp"
#include "composable_function_objects/material/StressTranslator.hpp"
#include "composable_function_objects/shape_function_operations/GeneralStressDivergence.hpp"
#include "elliptic/finite_deformation_mechanics/HyperElastostaticResidual_decl.hpp"
#include "utilities/ProblemDataParsingUtilities.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
template <typename EvaluationType, typename IndicatorFunctionType>
HyperElastostaticResidual<EvaluationType, IndicatorFunctionType>::HyperElastostaticResidual(
    const Plato::SpatialDomain& aSpatialDomain,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aProblemParams,
    Teuchos::ParameterList& aPenaltyParams)
    : FunctionBaseType(aSpatialDomain, aDataMap),
      mIndicatorFunction(aPenaltyParams),
      mApplyWeighting(mIndicatorFunction),
      mBodyLoads(utilities::get_body_loads<EvaluationType, ElementType>(aProblemParams)),
      mBoundaryLoads(utilities::get_boundary_loads<ElementType>(aProblemParams, "Natural Boundary Conditions")),
      mNeoHookeanParameters{composable_function_objects::material::get_neo_hookean_parameters(
          aProblemParams.sublist("Material Models").sublist(aSpatialDomain.getMaterialName()))},
      mPlotTable{utilities::get_plot_table(aProblemParams.sublist("Elliptic"))}
{
    utilities::get_displacement_dof_names(mNumSpatialDims, mDofNames);
}

template <typename EvaluationType, typename IndicatorFunctionType>
Plato::Solutions HyperElastostaticResidual<EvaluationType, IndicatorFunctionType>::getSolutionStateOutputData(
    const Plato::Solutions& aSolutions) const
{
    // no scaling, addition, or removal of data necessary for this physics.
    return aSolutions;
}

template <typename EvaluationType, typename IndicatorFunctionType>
void HyperElastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate(
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
{
    using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
    namespace kinematics = plato::composable_function_objects::kinematics;
    namespace material = plato::composable_function_objects::material;
    namespace shape_function_operations = plato::composable_function_objects::shape_function_operations;

    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    kinematics::DeformationGradient<ElementType> tComputeDeformationGradient;
    material::NeoHookeanModel tMaterialModel{mNeoHookeanParameters};
    material::StressTranslator<ResultScalarType, ElementType::mNumVoigtTerms> tStressTranslator;
    shape_function_operations::GeneralStressDivergence<ElementType> tComputeStressDivergence;
    auto& tComputeScaleFactorFromControl = mApplyWeighting;

    const auto tCubPoints = ElementType::getCubPoints();
    const auto tCubWeights = ElementType::getCubWeights();

    const auto tNumPoints = tCubWeights.size();
    const auto tNumCells = mSpatialDomain.numCells();

    Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, ElementType::mNumVoigtTerms);
    Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);

    Kokkos::parallel_for(
        "compute stress divergence contribution",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType tCellOrdinal, const Plato::OrdinalType tGpOrdinal) {
            const auto tCubPoint = tCubPoints(tGpOrdinal);

            ConfigScalarType tVolume(0.0);
            Plato::Matrix<mNumNodesPerCell, mNumSpatialDims, ConfigScalarType> tShapeGradients;
            tComputeGradient(tCellOrdinal, tCubPoint, aConfig, tShapeGradients, tVolume);
            tVolume *= tCubWeights(tGpOrdinal);

            Plato::Matrix<mTensorDim, mTensorDim, StrainScalarType> tDeformationGradient(0.0);
            tComputeDeformationGradient(tCellOrdinal, tDeformationGradient, aState, tShapeGradients);

            Plato::Matrix<mTensorDim, mTensorDim, ResultScalarType> tStress(0.0);
            tMaterialModel.stress(tDeformationGradient, tStress);

            const auto tCauchyStress =
                tStressTranslator.cauchyStressFromFirstPiolaKirchhoffStress(tStress, tDeformationGradient);
            for (int i = 0; i < ElementType::mNumVoigtTerms; i++)
            {
                Kokkos::atomic_add(&tCellStress(tCellOrdinal, i), tVolume * tCauchyStress(i));
            }
            Kokkos::atomic_add(&tCellVolume(tCellOrdinal), tVolume);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            ResultScalarType tScaleFactor{1.0};
            tComputeScaleFactorFromControl(tCellOrdinal, aControl, tBasisValues, tScaleFactor);
            Plato::Matrix<mTensorDim, mTensorDim, ResultScalarType> tStressScaled = Plato::times(tScaleFactor, tStress);

            tComputeStressDivergence(tCellOrdinal, aResult, tStressScaled, tShapeGradients, tVolume);
        });

    if (mBodyLoads.has_value())
    {
        mBodyLoads->get(mSpatialDomain, aState, aControl, aConfig, aResult, -1.0);
    }

    Kokkos::parallel_for(
        "compute averaged cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal) {
            for (Plato::OrdinalType i = 0; i < ElementType::mNumVoigtTerms; i++)
            {
                tCellStress(iCellOrdinal, i) /= tCellVolume(iCellOrdinal);
            }
        });

    if (std::count(mPlotTable.begin(), mPlotTable.end(), "stress"))
    {
        Plato::toMap(mDataMap, tCellStress, "stress", mSpatialDomain);
    }
}

template <typename EvaluationType, typename IndicatorFunctionType>
void HyperElastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate_boundary(
    const Plato::SpatialModel& aSpatialModel,
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
{
    if (mBoundaryLoads.has_value())
    {
        mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0, aTimeStep);
    }
}

template <typename EvaluationType, typename IndicatorFunctionType>
void HyperElastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate_contact(
    const Plato::SpatialModel& aSpatialModel,
    const std::string& aSideSet,
    Teuchos::RCP<Plato::Contact::AbstractSurfaceDisplacement<EvaluationType>> aComputeSurfaceDisp,
    Teuchos::RCP<Plato::Contact::AbstractContactForce<EvaluationType>> aComputeContactForce,
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
{
}
}  // namespace plato::elliptic::finite_deformation_mechanics
#endif
