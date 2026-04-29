#pragma once

#include "linear_algebra/PlatoMathHelpers.hpp"
#include "linear_algebra/ScalarGrad.hpp"
#include "local_operations/differential/GeneralFluxDivergence.hpp"
#include "local_operations/differential/GradientMatrix.hpp"
#include "local_operations/mapping/InterpolateFromNodal.hpp"
#include "local_operations/mapping/ProjectToNode.hpp"
#include "material/ThermalFlux.hpp"
#include "problem/ThermalContent.hpp"
#include "problem/parabolic/ParsingUtilities.hpp"

namespace Plato
{

namespace Parabolic
{

/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
HeatEquationResidual<EvaluationType, IndicatorFunctionType>::HeatEquationResidual(
    const plato::domain::SpatialDomain& aSpatialDomain,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aProblemParameters,
    Teuchos::ParameterList& aPenaltyParameters)
    : FunctionBaseType(aSpatialDomain, aDataMap),
      mIndicatorFunction(aPenaltyParameters),
      mApplyFluxWeighting(mIndicatorFunction),
      mApplyMassWeighting(mIndicatorFunction),
      mBoundaryLoads(nullptr)
/**************************************************************************/
{
    // obligatory: define dof names in order
    mDofNames.push_back("temperature");
    mDofDotNames.push_back("temperature rate");

    {
        Plato::ThermalConductionModelFactory<mNumSpatialDims> mmfactory(aProblemParameters);
        mThermalConductivityMaterialModel = mmfactory.create(aSpatialDomain.materialName());
    }

    {
        Plato::ThermalMassModelFactory<mNumSpatialDims> mmfactory(aProblemParameters);
        mThermalMassMaterialModel = mmfactory.create(aSpatialDomain.materialName());
    }

    plato::parabolic::check_time_step(aProblemParameters, aSpatialDomain);

    // parse boundary Conditions
    //
    if (aProblemParameters.isSublist("Natural Boundary Conditions"))
    {
        mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType, mNumDofsPerNode>>(
            aProblemParameters.sublist("Natural Boundary Conditions"));
    }
}

template <typename EvaluationType, typename IndicatorFunctionType>
Plato::Solutions HeatEquationResidual<EvaluationType, IndicatorFunctionType>::getSolutionStateOutputData(
    const Plato::Solutions& aSolutions) const
{
    return aSolutions;
}

/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void HeatEquationResidual<EvaluationType, IndicatorFunctionType>::evaluate(
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<StateDotScalarType>& aStateDot,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
/**************************************************************************/
{
    using GradScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

    auto tNumCells = mSpatialDomain.numCells();

    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    Plato::ScalarGrad<ElementType> tScalarGrad;
    Plato::ThermalFlux<ElementType> tThermalFlux(mThermalConductivityMaterialModel);
    Plato::GeneralFluxDivergence<ElementType> tFluxDivergence;
    Plato::ProjectToNode<ElementType> tProjectThermalEnergyRate;

    Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode> tInterpolateFromNodal;

    Plato::ThermalContent<mNumSpatialDims> tThermalContent(mThermalMassMaterialModel);

    auto tCubPoints = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints = tCubWeights.size();

    auto& tApplyFluxWeighting = mApplyFluxWeighting;
    auto& tApplyMassWeighting = mApplyMassWeighting;
    Kokkos::parallel_for(
        "compute residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal) {
            ConfigScalarType tVolume(0.0);

            Plato::Matrix<mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType> tGradient;

            Plato::Array<ElementType::mNumSpatialDims, GradScalarType> tGrad(0.0);
            Plato::Array<ElementType::mNumSpatialDims, ResultScalarType> tFlux(0.0);

            auto tCubPoint = tCubPoints(iGpOrdinal);
            auto tBasisValues = ElementType::basisValues(tCubPoint);

            tComputeGradient(iCellOrdinal, tCubPoint, aConfig, tGradient, tVolume);
            tVolume *= tCubWeights(iGpOrdinal);

            // compute temperature gradient
            //
            tScalarGrad(iCellOrdinal, tGrad, aState, tGradient);

            // compute flux
            //
            StateScalarType tTemperature = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aState);
            tThermalFlux(tFlux, tGrad, tTemperature);

            // apply weighting
            //
            tApplyFluxWeighting(iCellOrdinal, aControl, tBasisValues, tFlux);

            // compute stress divergence
            //
            tFluxDivergence(iCellOrdinal, aResult, tFlux, tGradient, tVolume, -1.0);

            // compute temperature at gausspoints
            //
            StateDotScalarType tTemperatureRate = tInterpolateFromNodal(iCellOrdinal, tBasisValues, aStateDot);

            // compute the time rate of internal thermal energy
            //
            ResultScalarType tThermalEnergyRate(0.0);
            tThermalContent(tThermalEnergyRate, tTemperatureRate, tTemperature);

            // apply weighting
            //
            tApplyMassWeighting(iCellOrdinal, aControl, tBasisValues, tThermalEnergyRate);

            // project to nodes
            //
            tProjectThermalEnergyRate(iCellOrdinal, tVolume, tBasisValues, tThermalEnergyRate, aResult);
        });
}
/**************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
void HeatEquationResidual<EvaluationType, IndicatorFunctionType>::evaluate_boundary(
    const plato::domain::SpatialModel& aSpatialModel,
    const Plato::ScalarMultiVectorT<StateScalarType>& aState,
    const Plato::ScalarMultiVectorT<StateDotScalarType>& aStateDot,
    const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
    const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
    Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
    Plato::Scalar aTimeStep) const
/**************************************************************************/
{
    if (mBoundaryLoads != nullptr)
    {
        mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
    }
}
}  // namespace Parabolic

}  // namespace Plato
