#pragma once

#include "boundary_conditions/ProblemDataParsingUtilities.hpp"
#include "core_types/FadTypes.hpp"
#include "core_types/PlatoTypes.hpp"
#include "domain/ToMap.hpp"
#include "domain/contact/IntegrateContactForce.hpp"
#include "local_operations/constitutive/LinearStress.hpp"
#include "local_operations/constitutive/VonMisesYieldFunction.hpp"
#include "local_operations/differential/GeneralStressDivergence.hpp"
#include "local_operations/differential/GradientMatrix.hpp"
#include "local_operations/kinematics/SmallStrain.hpp"
#include "problem/elliptic/ElastostaticResidual_decl.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
/**
 * \brief Constructor
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 * \param [in] aDataMap Plato Analyze database
 * \param [in] aProblemParams input parameters for overall problem
 * \param [in] aPenaltyParams input parameters for penalty function
 **********************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
ElastostaticResidual<EvaluationType, IndicatorFunctionType>::ElastostaticResidual(
    const plato::domain::SpatialDomain& aSpatialDomain,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aProblemParams,
    Teuchos::ParameterList& aPenaltyParams)
    : FunctionBaseType(aSpatialDomain, aDataMap),
      mIndicatorFunction(aPenaltyParams),
      mApplyWeighting(mIndicatorFunction),
      mBodyLoads(plato::utilities::get_body_loads<EvaluationType, ElementType>(aProblemParams)),
      mBoundaryLoads(plato::utilities::get_boundary_loads<ElementType>(aProblemParams, "Natural Boundary Conditions")),
      mPlotTable{plato::utilities::get_plot_table(aProblemParams.sublist("Elliptic"))}
{
    plato::utilities::get_displacement_dof_names(mNumSpatialDims, mDofNames);

    // create material model and get stiffness
    //
    Plato::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
    mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.materialName());

    // parse cell problem forcing
    //
    if (aProblemParams.isSublist("Cell Problem Forcing"))
    {
        Plato::OrdinalType tColumnIndex =
            aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
        mCellForcing.setCellStiffness(mMaterialModel->getStiffnessMatrix());
        mCellForcing.setColumnIndex(tColumnIndex);
    }
}

/****************************************************************************/
/**
 * \brief Pure virtual function to get output solution data
 * \param [in] state solution database
 * \return output state solution database
 ********************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
Plato::Solutions ElastostaticResidual<EvaluationType, IndicatorFunctionType>::getSolutionStateOutputData(
    const Plato::Solutions& aSolutions) const
{
    // No scaling, addition, or removal of data necessary for this physics.
    return aSolutions;
}

/******************************************************************************/
/**
 * \brief Evaluate vector function
 *
 * \param [in] aState 2D array with state variables (C,DOF)
 * \param [in] aControl 2D array with control variables (C,N)
 * \param [in] aConfig 3D array with control variables (C,N,D)
 * \param [in] aResult 1D array with control variables (C,DOF)
 * \param [in] aTimeStep current time step
 *
 * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
 * N = number of nodes per cell, D = spatial dimensions
 **********************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void ElastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate(
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
{
    using StrainScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
    namespace shape_function_operations = plato::composable_function_objects::shape_function_operations;

    auto tNumCells = mSpatialDomain.numCells();

    Plato::ComputeGradientMatrix<ElementType> computeGradient;
    Plato::SmallStrain<ElementType> computeVoigtStrain;
    shape_function_operations::GeneralStressDivergence<ElementType> computeStressDivergence;

    Plato::LinearStress<EvaluationType, ElementType> computeVoigtStress(mMaterialModel);

    Plato::ScalarVectorT<ConfigScalarType> tCellVolume("volume", tNumCells);

    Plato::ScalarMultiVectorT<StrainScalarType> tCellStrain("strain", tNumCells, mNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResultScalarType> tCellStress("stress", tNumCells, mNumVoigtTerms);

    auto tCubPoints = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints = tCubWeights.size();

    auto& applyWeighting = mApplyWeighting;
    auto& cellForcing = mCellForcing;

    Kokkos::parallel_for(
        "compute stress", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal) {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumVoigtTerms, StrainScalarType> tStrain(0.0);
            Plato::Array<ElementType::mNumVoigtTerms, ResultScalarType> tStress(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);

            computeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);

            computeVoigtStrain(iCellOrdinal, tStrain, aState, tGradient);

            computeVoigtStress(tStress, tStrain);

            cellForcing(tStress);

            tVolume *= tCubWeights(iGpOrdinal);

            auto tBasisValues = ElementType::basisValues(tCubPoint);
            applyWeighting(iCellOrdinal, aControl, tBasisValues, tStress);

            computeStressDivergence(iCellOrdinal, aResult, tStress, tGradient, tVolume);

            for (int i = 0; i < ElementType::mNumVoigtTerms; i++)
            {
                Kokkos::atomic_add(&tCellStrain(iCellOrdinal, i), tVolume * tStrain(i));
                Kokkos::atomic_add(&tCellStress(iCellOrdinal, i), tVolume * tStress(i));
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
        });

    if (mBodyLoads.has_value())
    {
        mBodyLoads->get(mSpatialDomain, aState, aControl, aConfig, aResult, -1.0);
    }

    if (std::count(mPlotTable.begin(), mPlotTable.end(), "strain"))
    {
        Plato::toMap(mDataMap, tCellStrain, "strain", mSpatialDomain);
    }
    if (std::count(mPlotTable.begin(), mPlotTable.end(), "stress"))
    {
        Plato::toMap(mDataMap, tCellStress, "stress", mSpatialDomain);
    }
    if (std::count(mPlotTable.begin(), mPlotTable.end(), "Vonmises"))
    {
        this->outputVonMises(tCellStress, mSpatialDomain);
    }
}

/******************************************************************************/
/**
 * \brief Evaluate vector function
 *
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aState 2D array with state variables (C,DOF)
 * \param [in] aControl 2D array with control variables (C,N)
 * \param [in] aConfig 3D array with control variables (C,N,D)
 * \param [in] aResult 1D array with control variables (C,DOF)
 * \param [in] aTimeStep current time step
 *
 * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
 * N = number of nodes per cell, D = spatial dimensions
 **********************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void ElastostaticResidual<EvaluationType, IndicatorFunctionType>::evaluate_boundary(
    const plato::domain::SpatialModel& aSpatialModel,
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
{
    if (mBoundaryLoads.has_value())
    {
        mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
    }
}

/**********************************************************************/
/**
 * \brief Compute Von Mises stress field and copy data into output data map
 * \param [in] aCauchyStress Cauchy stress tensor
 **************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void ElastostaticResidual<EvaluationType, IndicatorFunctionType>::outputVonMises(
    const Plato::ScalarMultiVectorT<ResultScalarType>& aCauchyStress,
    const plato::domain::SpatialDomain& aSpatialDomain) const
{
    auto tNumCells = aSpatialDomain.numCells();
    Plato::VonMisesYieldFunction<mNumSpatialDims, mNumVoigtTerms> tComputeVonMises;
    Plato::ScalarVectorT<ResultScalarType> tVonMises("Von Mises", tNumCells);
    Kokkos::parallel_for(
        "Compute VonMises Stress", Kokkos::RangePolicy<>(0, tNumCells),
        KOKKOS_LAMBDA(const Plato::OrdinalType& aCellOrdinal) {
            ResultScalarType tCellVonMises(0);
            tComputeVonMises(aCellOrdinal, aCauchyStress, tCellVonMises);
            tVonMises(aCellOrdinal) = tCellVonMises;
        });

    Plato::toMap(mDataMap, tVonMises, "Vonmises", aSpatialDomain);
}

}  // namespace Elliptic

}  // namespace Plato
