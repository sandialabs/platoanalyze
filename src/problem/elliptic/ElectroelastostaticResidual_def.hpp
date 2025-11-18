#pragma once

#include "boundary_conditions/ProblemDataParsingUtilities.hpp"
#include "core_types/FadTypes.hpp"
#include "core_types/PlatoTypes.hpp"
#include "domain/ToMap.hpp"
#include "linear_algebra/BLAS2.hpp"
#include "local_operations/differential/GeneralFluxDivergence.hpp"
#include "local_operations/differential/GeneralStressDivergence.hpp"
#include "local_operations/differential/GradientMatrix.hpp"
#include "material/EMKinematics.hpp"
#include "material/EMKinetics.hpp"
#include "problem/elliptic/ElectroelastostaticResidual_decl.hpp"

namespace Plato
{

namespace Elliptic
{

/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::ElectroelastostaticResidual(
    const Plato::SpatialDomain& aSpatialDomain,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aProblemParams,
    Teuchos::ParameterList& aPenaltyParams)
    : FunctionBaseType(aSpatialDomain, aDataMap),
      mIndicatorFunction(aPenaltyParams),
      mApplyStressWeighting(mIndicatorFunction),
      mApplyEDispWeighting(mIndicatorFunction),
      mBodyLoads(plato::utilities::get_body_loads<EvaluationType, ElementType>(aProblemParams)),
      mBoundaryLoads(plato::utilities::get_boundary_loads<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>(
          aProblemParams, "Mechanical Natural Boundary Conditions")),
      mBoundaryCharges(plato::utilities::get_boundary_loads<ElementType, NElecDims, mNumDofsPerNode, EDofOffset>(
          aProblemParams, "Electrical Natural Boundary Conditions")),
      mPlottable{plato::utilities::get_plot_table(aProblemParams.sublist("Electroelastostatics"))}
/**************************************************************************/
{
    plato::utilities::get_displacement_dof_names(mNumSpatialDims, mDofNames);
    mDofNames.push_back("electric potential");

    // create material model and get stiffness
    //
    Plato::ElectroelasticModelFactory<mNumSpatialDims> mmfactory(aProblemParams);
    mMaterialModel = mmfactory.create(mSpatialDomain.getMaterialName());
}

/****************************************************************************/
/**
 * \brief Pure virtual function to get output solution data
 * \param [in] state solution database
 * \return output state solution database
 ********************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
Plato::Solutions ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::getSolutionStateOutputData(
    const Plato::Solutions& aSolutions) const
{
    return aSolutions;
}

/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate(
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
    Plato::EMKinematics<ElementType> kinematics;
    Plato::EMKinetics<ElementType> kinetics(mMaterialModel);

    shape_function_operations::GeneralStressDivergence<ElementType, mNumDofsPerNode, MDofOffset> stressDivergence;
    Plato::GeneralFluxDivergence<ElementType, mNumDofsPerNode, EDofOffset> edispDivergence;

    Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight", tNumCells);

    Plato::ScalarMultiVectorT<GradScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
    Plato::ScalarMultiVectorT<GradScalarType> tCellEField("efield", tNumCells, mNumSpatialDims);

    Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResultScalarType> tCellEDisp("edisp", tNumCells, mNumSpatialDims);

    auto tCubPoints = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints = tCubWeights.size();

    auto& applyStressWeighting = mApplyStressWeighting;
    auto& applyEDispWeighting = mApplyEDispWeighting;
    Kokkos::parallel_for(
        "compute element state", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal) {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, GradScalarType> tStrain(0.0);
            Plato::Array<ElementType::mNumSpatialDims, GradScalarType> tEField(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);
            Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tEDisp(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

            tVolume *= tCubWeights(iGpOrdinal);

            // compute strain and electric field
            //
            kinematics(iCellOrdinal, tStrain, tEField, aState, tGradient);

            // compute stress and electric displacement
            //
            kinetics(tStress, tEDisp, tStrain, tEField);

            // apply weighting
            //
            auto tBasisValues = ElementType::basisValues(tCubPoint);
            applyStressWeighting(iCellOrdinal, aControl, tBasisValues, tStress);
            applyEDispWeighting(iCellOrdinal, aControl, tBasisValues, tEDisp);

            // compute divergence
            //
            stressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);
            edispDivergence(iCellOrdinal, aResult, tEDisp, tGradient, tVolume);

            for (int i = 0; i < ElementType::mNumVoigtTerms; i++)
            {
                Kokkos::atomic_add(&tCellStrain(iCellOrdinal, i), tVolume * tStrain(i));
                Kokkos::atomic_add(&tCellStress(iCellOrdinal, i), tVolume * tStress(i));
            }
            for (int i = 0; i < ElementType::mNumSpatialDims; i++)
            {
                Kokkos::atomic_add(&tCellEField(iCellOrdinal, i), tVolume * tEField(i));
                Kokkos::atomic_add(&tCellEDisp(iCellOrdinal, i), tVolume * tEDisp(i));
            }
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
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
                tCellEField(iCellOrdinal, i) /= tCellVolume(iCellOrdinal);
                tCellEDisp(iCellOrdinal, i) /= tCellVolume(iCellOrdinal);
            }
        });

    if (mBodyLoads.has_value())
    {
        mBodyLoads->get(mSpatialDomain, aState, aControl, aConfig, aResult, -1.0);
    }

    if (std::count(mPlottable.begin(), mPlottable.end(), "strain"))
        toMap(mDataMap, tCellStrain, "strain", mSpatialDomain);
    if (std::count(mPlottable.begin(), mPlottable.end(), "efield"))
        toMap(mDataMap, tCellEField, "efield", mSpatialDomain);
    if (std::count(mPlottable.begin(), mPlottable.end(), "stress"))
        toMap(mDataMap, tCellStress, "stress", mSpatialDomain);
    if (std::count(mPlottable.begin(), mPlottable.end(), "edisp")) toMap(mDataMap, tCellEDisp, "edisp", mSpatialDomain);
}
/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void ElectroelastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate_boundary(
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

    if (mBoundaryCharges.has_value())
    {
        mBoundaryCharges->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
    }
}

}  // namespace Elliptic

}  // namespace Plato
