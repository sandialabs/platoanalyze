#pragma once

#include "boundary_conditions/ProblemDataParsingUtilities.hpp"
#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/BLAS2.hpp"
#include "local_operations/PressureDivergence.hpp"
#include "local_operations/differential/GeneralFluxDivergence.hpp"
#include "local_operations/differential/GeneralStressDivergence.hpp"
#include "local_operations/differential/GradientMatrix.hpp"
#include "local_operations/mapping/InterpolateFromNodal.hpp"
#include "local_operations/mapping/ProjectToNode.hpp"
#include "problem/elliptic/stabilized/TMKinematics.hpp"
#include "problem/elliptic/stabilized/TMKinetics.hpp"

namespace Plato
{

namespace Stabilized
{

/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
ThermoelastostaticResidual<EvaluationType, IndicatorFunctionType>::ThermoelastostaticResidual(
    const Plato::SpatialDomain& aSpatialDomain,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aProblemParams,
    Teuchos::ParameterList& aPenaltyParams)
    : FunctionBaseType(aSpatialDomain, aDataMap),
      mIndicatorFunction(aPenaltyParams),
      mApplyTensorWeighting(mIndicatorFunction),
      mApplyVectorWeighting(mIndicatorFunction),
      mApplyScalarWeighting(mIndicatorFunction),
      mBodyLoads(plato::utilities::get_body_loads<EvaluationType, ElementType>(aProblemParams)),
      mBoundaryLoads(plato::utilities::get_boundary_loads<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>(
          aProblemParams, "Mechanical Natural Boundary Conditions")),
      mBoundaryFluxes(plato::utilities::get_boundary_loads<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>(
          aProblemParams, "Thermal Natural Boundary Conditions"))
/**************************************************************************/
{
    plato::utilities::get_displacement_dof_names(mNumSpatialDims, mDofNames);
    mDofNames.push_back("pressure");
    mDofNames.push_back("temperature");

    // create material model and get stiffness
    //
    Plato::LinearThermoelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
    mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
}

/****************************************************************************/
/**
 * \brief Pure virtual function to get output solution data
 * \param [in] state solution database
 * \return output state solution database
 ********************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
Plato::Solutions ThermoelastostaticResidual<EvaluationType, IndicatorFunctionType>::getSolutionStateOutputData(
    const Plato::Solutions& aSolutions) const
{
    return aSolutions;
}

/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void ThermoelastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate(
    const Plato::ScalarMultiVectorT<StateScalarType>& aStateWS,
    const Plato::ScalarMultiVectorT<NodeStateScalarType>& aPGradWS,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControlWS,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfigWS,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResultWS,
    Plato::Scalar aTimeStep) const
/**************************************************************************/
{
    namespace shape_function_operations = plato::composable_function_objects::shape_function_operations;

    auto tNumCells = mSpatialDomain.numCells();

    using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

    Plato::ComputeGradientMatrix<ElementType> computeGradient;
    Plato::Stabilized::TMKinematics<ElementType> kinematics;
    Plato::Stabilized::TMKinetics<ElementType> kinetics(mMaterialModel);

    Plato::InterpolateFromNodal<ElementType, mNumSpatialDims, 0, mNumSpatialDims> interpolatePGradFromNodal;
    Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, PDofOffset> interpolatePressureFromNodal;
    Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> interpolateTemperatureFromNodal;

    Plato::GeneralFluxDivergence<ElementType, mNumDofsPerNode, TDofOffset> fluxDivergence;
    Plato::GeneralFluxDivergence<ElementType, mNumDofsPerNode, PDofOffset> stabDivergence;
    shape_function_operations::GeneralStressDivergence<ElementType, mNumDofsPerNode, MDofOffset> stressDivergence;
    Plato::ProjectToNode<ElementType, mNumDofsPerNode, PDofOffset> projectVolumeStrain;

    Plato::PressureDivergence<ElementType, mNumDofsPerNode> pressureDivergence;

    auto tCubPoints = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints = tCubWeights.size();

    auto& applyTensorWeighting = mApplyTensorWeighting;
    auto& applyVectorWeighting = mApplyVectorWeighting;
    auto& applyScalarWeighting = mApplyScalarWeighting;

    Kokkos::parallel_for(
        "compute residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal) {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            // compute gradient operator and cell volume
            //
            auto tCubPoint = tCubPoints(iGpOrdinal);
            computeGradient(iCellOrdinal, tCubPoint, aConfigWS, tGradient, tVolume);
            tVolume *= tCubWeights(iGpOrdinal);

            // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
            //
            Plato::Array<mNumVoigtTerms, GradScalarType> tDGrad(0.0);
            Plato::Array<mNumSpatialDims, GradScalarType> tPGrad(0.0);
            Plato::Array<mNumSpatialDims, GradScalarType> tTGrad(0.0);
            kinematics(iCellOrdinal, tDGrad, tPGrad, tTGrad, aStateWS, tGradient);

            // interpolate projected PGrad, pressure, and temperature to gauss point
            //
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            Plato::Array<mNumSpatialDims, NodeStateScalarType> tProjectedPGrad(0.0);
            interpolatePGradFromNodal(iCellOrdinal, tBasisValues, aPGradWS, tProjectedPGrad);

            ResultScalarType tPressure;
            interpolatePressureFromNodal(iCellOrdinal, tBasisValues, aStateWS, tPressure);

            ResultScalarType tTemperature;
            interpolateTemperatureFromNodal(iCellOrdinal, tBasisValues, aStateWS, tTemperature);

            // compute the constitutive response
            //
            ResultScalarType tVolStrain(0.0);
            Plato::Array<mNumSpatialDims, ResultScalarType> tCellStab(0.0);
            Plato::Array<mNumSpatialDims, ResultScalarType> tTFlux(0.0);
            Plato::Array<mNumVoigtTerms, ResultScalarType> tDevStress(0.0);
            kinetics(tVolume, tProjectedPGrad, tDGrad, tPGrad, tTGrad, tTemperature, tPressure, tDevStress, tVolStrain,
                     tTFlux, tCellStab);

            // apply weighting
            //
            applyTensorWeighting(iCellOrdinal, aControlWS, tBasisValues, tDevStress);
            applyVectorWeighting(iCellOrdinal, aControlWS, tBasisValues, tCellStab);
            applyVectorWeighting(iCellOrdinal, aControlWS, tBasisValues, tTFlux);
            applyScalarWeighting(iCellOrdinal, aControlWS, tBasisValues, tPressure);
            applyScalarWeighting(iCellOrdinal, aControlWS, tBasisValues, tVolStrain);

            // compute divergence
            //
            stressDivergence(iCellOrdinal, aResultWS, tDevStress, tGradient, tVolume);
            pressureDivergence(iCellOrdinal, aResultWS, tPressure, tGradient, tVolume);
            stabDivergence(iCellOrdinal, aResultWS, tCellStab, tGradient, tVolume, -1.0);
            fluxDivergence(iCellOrdinal, aResultWS, tTFlux, tGradient, tVolume);

            projectVolumeStrain(iCellOrdinal, tVolume, tBasisValues, tVolStrain, aResultWS);
        });

    if (mBodyLoads.has_value())
    {
        mBodyLoads->get(mSpatialDomain, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0);
    }
}
/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void ThermoelastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate_boundary(
    const Plato::SpatialModel& aSpatialModel,
    const Plato::ScalarMultiVectorT<StateScalarType>& aStateWS,
    const Plato::ScalarMultiVectorT<NodeStateScalarType>& aPGradWS,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControlWS,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfigWS,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResultWS,
    Plato::Scalar aTimeStep) const
/**************************************************************************/
{
    if (mBoundaryLoads.has_value())
    {
        mBoundaryLoads->get(aSpatialModel, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0);
    }

    if (mBoundaryFluxes.has_value())
    {
        mBoundaryFluxes->get(aSpatialModel, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0);
    }
}
}  // namespace Stabilized
}  // namespace Plato
