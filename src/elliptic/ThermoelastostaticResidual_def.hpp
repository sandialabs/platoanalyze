#pragma once

#include "BLAS2.hpp"
#include "FadTypes.hpp"
#include "GeneralFluxDivergence.hpp"
#include "GradientMatrix.hpp"
#include "InterpolateFromNodal.hpp"
#include "PlatoTypes.hpp"
#include "TMKinematics.hpp"
#include "TMKineticsFactory.hpp"
#include "ToMap.hpp"
#include "composable_function_objects/shape_function_operations/GeneralStressDivergence.hpp"
#include "elliptic/ThermoelastostaticResidual_decl.hpp"
#include "utilities/ProblemDataParsingUtilities.hpp"

namespace Plato
{

namespace Elliptic
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
      mApplyStressWeighting(mIndicatorFunction),
      mApplyFluxWeighting(mIndicatorFunction),
      mBodyLoads(plato::utilities::get_body_loads<EvaluationType, ElementType>(aProblemParams)),
      mBoundaryLoads(plato::utilities::get_boundary_loads<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>(
          aProblemParams, "Mechanical Natural Boundary Conditions")),
      mBoundaryFluxes(plato::utilities::get_boundary_loads<ElementType, NThrmDims, mNumDofsPerNode, TDofOffset>(
          aProblemParams, "Thermal Natural Boundary Conditions")),
      mPlottable{plato::utilities::get_plot_table(aProblemParams.sublist("Elliptic"))}
/**************************************************************************/
{
    // obligatory: define dof names in order
    plato::utilities::get_displacement_dof_names(mNumSpatialDims, mDofNames);
    mDofNames.push_back("temperature");

    // create material model and get stiffness
    //
    Plato::ThermoelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
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
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
/**************************************************************************/
{
    namespace shape_function_operations = plato::composable_function_objects::shape_function_operations;

    auto tNumCells = mSpatialDomain.numCells();

    using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

    Plato::ComputeGradientMatrix<ElementType> computeGradient;
    Plato::TMKinematics<ElementType> kinematics;

    Plato::TMKineticsFactory<EvaluationType, ElementType> tTMKineticsFactory;
    auto tTMKinetics = tTMKineticsFactory.create(mMaterialModel, mSpatialDomain, mDataMap);

    shape_function_operations::GeneralStressDivergence<ElementType, mNumDofsPerNode, MDofOffset> stressDivergence;
    Plato::GeneralFluxDivergence<ElementType, mNumDofsPerNode, TDofOffset> fluxDivergence;

    Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

    Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);

    Plato::ScalarMultiVectorT<GradScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
    Plato::ScalarMultiVectorT<GradScalarType> tCellTgrad("tgrad", tNumCells, mNumSpatialDims);

    Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResultScalarType> tCellFlux("flux", tNumCells, mNumSpatialDims);

    auto tCubPoints = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints = tCubWeights.size();

    Plato::ScalarArray3DT<ResultScalarType> tStress("stress", tNumCells, tNumPoints, mNumVoigtTerms);
    Plato::ScalarArray3DT<ResultScalarType> tFlux("flux", tNumCells, tNumPoints, mNumSpatialDims);
    Plato::ScalarArray3DT<GradScalarType> tStrain("strain", tNumCells, tNumPoints, mNumVoigtTerms);
    Plato::ScalarArray3DT<GradScalarType> tTGrad("tgrad", tNumCells, tNumPoints, mNumSpatialDims);

    Plato::ScalarArray4DT<ConfigScalarType> tGradient("gradient", tNumCells, tNumPoints, mNumNodesPerCell,
                                                      mNumSpatialDims);

    Plato::ScalarMultiVectorT<ConfigScalarType> tVolume("volume", tNumCells, tNumPoints);
    Plato::ScalarMultiVectorT<StateScalarType> tTemperature("temperature", tNumCells, tNumPoints);

    auto& applyStressWeighting = mApplyStressWeighting;
    auto& applyFluxWeighting = mApplyFluxWeighting;
    Kokkos::parallel_for(
        "compute element kinematics", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal) {
            auto tCubPoint = tCubPoints(iGpOrdinal);

            computeGradient(iCellOrdinal, iGpOrdinal, tCubPoint, aConfig, tGradient, tVolume);

            tVolume(iCellOrdinal, iGpOrdinal) *= tCubWeights(iGpOrdinal);

            // compute strain and electric field
            //
            kinematics(iCellOrdinal, iGpOrdinal, tStrain, tTGrad, aState, tGradient);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            tTemperature(iCellOrdinal, iGpOrdinal) = interpolateFromNodal(iCellOrdinal, tBasisValues, aState);
        });

    // compute element state
    (*tTMKinetics)(tStress, tFlux, tStrain, tTGrad, tTemperature, aControl);

    Kokkos::parallel_for(
        "compute divergence", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal) {
            // apply weighting
            //
            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            applyStressWeighting(iCellOrdinal, iGpOrdinal, aControl, tBasisValues, tStress);
            applyFluxWeighting(iCellOrdinal, iGpOrdinal, aControl, tBasisValues, tFlux);

            // compute divergence
            //
            stressDivergence(iCellOrdinal, iGpOrdinal, aResult, tStress, tGradient, tVolume);
            fluxDivergence(iCellOrdinal, iGpOrdinal, aResult, tFlux, tGradient, tVolume);

            for (int i = 0; i < ElementType::mNumVoigtTerms; i++)
            {
                Kokkos::atomic_add(&tCellStrain(iCellOrdinal, i),
                                   tVolume(iCellOrdinal, iGpOrdinal) * tStrain(iCellOrdinal, iGpOrdinal, i));
                Kokkos::atomic_add(&tCellStress(iCellOrdinal, i),
                                   tVolume(iCellOrdinal, iGpOrdinal) * tStress(iCellOrdinal, iGpOrdinal, i));
            }
            for (int i = 0; i < ElementType::mNumSpatialDims; i++)
            {
                Kokkos::atomic_add(&tCellTgrad(iCellOrdinal, i),
                                   tVolume(iCellOrdinal, iGpOrdinal) * tTGrad(iCellOrdinal, iGpOrdinal, i));
                Kokkos::atomic_add(&tCellFlux(iCellOrdinal, i),
                                   tVolume(iCellOrdinal, iGpOrdinal) * tFlux(iCellOrdinal, iGpOrdinal, i));
            }
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume(iCellOrdinal, iGpOrdinal));
        });

    Kokkos::parallel_for(
        "compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal) {
            for (int i = 0; i < ElementType::mNumVoigtTerms; i++)
            {
                tCellStrain(iCellOrdinal, i) /= tCellVolume(iCellOrdinal);
                tCellStress(iCellOrdinal, i) /= tCellVolume(iCellOrdinal);
            }
            for (int i = 0; i < ElementType::mNumSpatialDims; i++)
            {
                tCellTgrad(iCellOrdinal, i) /= tCellVolume(iCellOrdinal);
                tCellFlux(iCellOrdinal, i) /= tCellVolume(iCellOrdinal);
            }
        });

    if (mBodyLoads.has_value())
    {
        mBodyLoads->get(mSpatialDomain, aState, aControl, aConfig, aResult, -1.0);
    }

    if (std::count(mPlottable.begin(), mPlottable.end(), "strain"))
        toMap(mDataMap, tCellStrain, "strain", mSpatialDomain);
    if (std::count(mPlottable.begin(), mPlottable.end(), "tgrad")) toMap(mDataMap, tCellTgrad, "tgrad", mSpatialDomain);
    if (std::count(mPlottable.begin(), mPlottable.end(), "stress"))
        toMap(mDataMap, tCellStress, "stress", mSpatialDomain);
    if (std::count(mPlottable.begin(), mPlottable.end(), "flux")) toMap(mDataMap, tCellFlux, "flux", mSpatialDomain);
}
/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void ThermoelastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate_boundary(
    const Plato::SpatialModel& aSpatialModel,
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
/**************************************************************************/
{
    if (mBoundaryLoads.has_value())
    {
        mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
    }

    if (mBoundaryFluxes.has_value())
    {
        mBoundaryFluxes->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
    }
}
}  // namespace Elliptic

}  // namespace Plato
