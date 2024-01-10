/*
 * InternalThermalForces.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "SpatialModel.hpp"
#include "UtilsTeuchos.hpp"
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/fluids/AbstractVolumeIntegrand.hpp"
#include "hyperbolic/fluids/FluidsUtils.hpp"
#include "hyperbolic/fluids/SimplexFluids.hpp"
#include "hyperbolic/fluids/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/fluids/EnergyConservationUtils.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \class InternalThermalForces
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Derived class responsible for the evaluation of the internal thermal
 *   forces. This implementation is used for forward simulations. In addition,
 *   this implementation can be used for level-set based topology optimization
 *   problems and parametric CAD shape optimization problems.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class InternalThermalForces : public Plato::AbstractVolumeIntegrand<PhysicsT,EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum degrees of freedom per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell; /*!< number of energy degrees of freedom per node */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum degrees of freedom per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode; /*!< number of energy degrees of freedom per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell; /*!< number of configuration degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */
    using CurVelT   = typename EvaluationT::CurrentMomentumScalarType; /*!< current velocity FAD evaluation type */
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType; /*!< current temperature FAD evaluation type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous temperature FAD evaluation type */

    using CurFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurTempT, ConfigT>; /*!< current flux FAD evaluation type */
    using PrevFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, ConfigT>; /*!< previous flux FAD evaluation type */
    using ConvectionT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, CurVelT, ConfigT>; /*!< convection FAD evaluation type */

    Plato::Scalar mArtificialDamping = 1.0; /*!< artificial temperature damping - damping is a byproduct from time integration scheme */
    Plato::Scalar mStabilizationMultiplier = 1.0; /*!< stabilization scalar multiplier */
    Plato::Scalar mEffectiveThermalProperty = 1.0; /*!< effective thermal property for diffusivity term */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    InternalThermalForces
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
        mDataMap(aDataMap),
        mSpatialDomain(aDomain),
        mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setAritificalDiffusiveDamping(aInputs);
        auto tMyMaterialName = mSpatialDomain.getMaterialName();
        mEffectiveThermalProperty = Plato::Fluids::calculate_effective_conductivity(tMyMaterialName, aInputs);
        mStabilizationMultiplier = Plato::Fluids::stabilization_constant("Energy Conservation", aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    ~InternalThermalForces(){}

    /***************************************************************************//**
     * \brief Evaluate internal thermal forces.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            ANALYZE_THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local data
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarVectorT<CurTempT> tCurTempGP("current temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<ConvectionT> tConvection("convection", tNumCells);

        Plato::ScalarMultiVectorT<CurFluxT> tCurThermalFlux("current thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevFluxT> tPrevThermalFlux("previous thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss points", tNumCells, mNumVelDofsPerNode);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tTheta           = mArtificialDamping;
        auto tEffectiveThermalProperty = mEffectiveThermalProperty;
        auto tStabilizationMultiplier = mStabilizationMultiplier;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for("energy conservation residual", Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) = tCellVolume(aCellOrdinal) * tCubWeight;

            // 1. add previous diffusive force contribution to residual, i.e. R -= (\theta_3-1) K T^n
            auto tMultiplier = (tTheta - static_cast<Plato::Scalar>(1));
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevTempWS, tPrevThermalFlux);
            Plato::blas2::scale<mNumSpatialDims>(aCellOrdinal, tEffectiveThermalProperty, tPrevThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tPrevThermalFlux, aResultWS, -tMultiplier);

            // 2. add current convective force contribution to residual, i.e. R += C(u^{n+1}) T^n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::Fluids::calculate_convective_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurVelGP, tPrevTempWS, tConvection);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tConvection, aResultWS, 1.0);

            // 3. add current diffusive force contribution to residual, i.e. R += \theta_3 K T^{n+1}
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurTempWS, tCurThermalFlux);
            Plato::blas2::scale<mNumSpatialDims>(aCellOrdinal, tEffectiveThermalProperty, tCurThermalFlux);
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tCurThermalFlux, aResultWS, tTheta);
            Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 5. add stabilizing convective force contribution to residual, i.e. R += \alpha_{stab} * C_u(u^{n+1}) T^n
            tMultiplier = tStabilizationMultiplier * static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tCurVelGP, tConvection, aResultWS, tMultiplier);
            
            // 6. add previous inertial force contribution to residual, i.e. R -= M T^n
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevTempGP, aResultWS, -1.0);

            // 7. add current inertial force contribution to residual, i.e. R += M T^{n+1}
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tCurTempWS, tCurTempGP);
            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurTempGP, aResultWS, 1.0);
        });
    }

private:
    /***************************************************************************//**
     * \brief Set artificial damping.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setAritificalDiffusiveDamping
    (Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mArtificialDamping = tTimeIntegration.get<Plato::Scalar>("Diffusive Damping", 1.0);
        }
    }
};
// class InternalThermalForces

}
// namespace Fluids

}
// namespace Plato

// ********************************************************************************************************************
// * Evaluation of internal thermal forces for density-based topology optimization in forced convection applications. *
// ********************************************************************************************************************
namespace Plato
{

namespace Fluids
{

namespace SIMP
{


/***************************************************************************//**
 * \class InternalThermalForces
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Derived class responsible for the evaluation of the internal thermal
 *   forces. This implementation is only used for density-based topology
 *   optimization problems.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class InternalThermalForces : public Plato::AbstractVolumeIntegrand<PhysicsT,EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell/element */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum degrees of freedom per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell; /*!< number of energy degrees of freedom per node */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum degrees of freedom per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode; /*!< number of energy degrees of freedom per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell; /*!< number of configuration degrees of freedom per cell */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */
    using CurVelT   = typename EvaluationT::CurrentMomentumScalarType; /*!< current momentum FAD evaluation type */
    using CurTempT  = typename EvaluationT::CurrentEnergyScalarType; /*!< current energy FAD evaluation type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous energy FAD evaluation type */

    using CurFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurTempT, ConfigT>; /*!< current flux FAD evaluation type */
    using PrevFluxT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, ConfigT>; /*!< previous flux FAD evaluation type */
    using ConvectionT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevTempT, CurVelT, ConfigT>; /*!< convection FAD evaluation type */

    Plato::Scalar mArtificialDamping = 1.0; /*!< artificial temperature damping - damping is a byproduct from time integration scheme */
    Plato::Scalar mStabilizationMultiplier = 0.0; /*!< stabilization scalar multiplier */
    Plato::Scalar mEffectiveThermalProperty = 1.0; /*!< effective thermal property for diffusivity term */
    Plato::Scalar mDiffusiveTermPenaltyExponent = 3.0; /*!< penalty model exponent used for diffusive term */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature integration rule */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    InternalThermalForces
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
        mDataMap(aDataMap),
        mSpatialDomain(aDomain),
        mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setPenaltyModelParameters(aInputs);
        this->setAritificalDiffusiveDamping(aInputs);

        auto tMyMaterialName = mSpatialDomain.getMaterialName();
        mEffectiveThermalProperty = Plato::Fluids::calculate_effective_conductivity(tMyMaterialName, aInputs);
        mStabilizationMultiplier = Plato::Fluids::stabilization_constant("Energy Conservation", aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    ~InternalThermalForces(){}

    /***************************************************************************//**
     * \brief Evaluate internal thermal forces. This implementation is only used for
     *   density-based topology optimization problems.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            ANALYZE_THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local arrays
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarVectorT<CurTempT> tCurTempGP("current temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss points", tNumCells);
        Plato::ScalarVectorT<ConvectionT> tConvection("convection", tNumCells);

        Plato::ScalarMultiVectorT<CurFluxT> tCurThermalFlux("current thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevFluxT> tPrevThermalFlux("previous thermal flux", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss points", tNumCells, mNumVelDofsPerNode);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));
        auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tCurTempWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurTempT>>(aWorkSets.get("current temperature"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to temporary local scope that device can access
        auto tArtificialDamping = mArtificialDamping;
        auto tStabilizationMultiplier = mStabilizationMultiplier;
        auto tEffectiveThermalProperty = mEffectiveThermalProperty;
        auto tDiffusiveTermPenaltyExponent = mDiffusiveTermPenaltyExponent;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for("energy conservation residual", Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
        {
            // compte cell volumes
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) = tCellVolume(aCellOrdinal) * tCubWeight;

            // 1. Penalize effective thermal property
            ControlT tPenalizedEffectiveThermalProperty = Plato::Fluids::penalized_effective_thermal_property<mNumNodesPerCell>
                (aCellOrdinal, tEffectiveThermalProperty, tDiffusiveTermPenaltyExponent, tControlWS);

            // 2. add current diffusive force contribution to residual, i.e. R += \theta_3 pi^{\alpha}(theta) K T^{n+1},
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurTempWS, tCurThermalFlux);
            ControlT tMultiplierControlOneT = tArtificialDamping * tPenalizedEffectiveThermalProperty;
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tCurThermalFlux, aResultWS, tMultiplierControlOneT);

            // 3. add previous diffusive force contribution to residual, i.e. R -= (\theta_3-1) pi^{\alpha}(theta) K T^n
            Plato::Fluids::calculate_flux<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevTempWS, tPrevThermalFlux);
            ControlT tMultiplierControlTwoT = (tArtificialDamping - static_cast<Plato::Scalar>(1.0)) * tPenalizedEffectiveThermalProperty;
            Plato::Fluids::calculate_flux_divergence<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCellVolume, tPrevThermalFlux, aResultWS, -tMultiplierControlTwoT);

            // 4. add current convective force contribution to residual, i.e. R += C(u^{n+1}) T^n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::Fluids::calculate_convective_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tCurVelGP, tPrevTempWS, tConvection);
            Plato::Fluids::integrate_scalar_field<mNumTempDofsPerCell>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tConvection, aResultWS, 1.0);
            Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 5. add stabilizing force contribution to residual, i.e. R += \alpha_{stab} * C_u(u^{n+1}) T^n
            Plato::Scalar tScalar = tStabilizationMultiplier * 
                static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_scalar_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tCurVelGP, tConvection, aResultWS, tScalar);

            // 6. add previous inertial force contribution to residual, i.e. R -= M T^n
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>(aCellOrdinal, tBasisFunctions, tCellVolume, tPrevTempGP, aResultWS, -1.0);

            // 7. add current inertial force contribution to residual, i.e. R += M T^{n+1}
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tCurTempWS, tCurTempGP);
            Plato::Fluids::integrate_scalar_field<mNumNodesPerCell>(aCellOrdinal, tBasisFunctions, tCellVolume, tCurTempGP, aResultWS, 1.0);
        });
    }

private:
    /***************************************************************************//**
     * \brief Set artificial diffusive damping.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setAritificalDiffusiveDamping
    (Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mArtificialDamping = tTimeIntegration.get<Plato::Scalar>("Diffusive Damping", 1.0);
        }
    }

    /***************************************************************************//**
     * \brief Set penalty parameters for density penalization model.
     * \param [in] aInputs input file metadata.
     ******************************************************************************/
    void setPenaltyModelParameters
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            ANALYZE_THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolicParamList = aInputs.sublist("Hyperbolic");

        if(tHyperbolicParamList.isSublist("Energy Conservation"))
        {
            auto tEnergyParamList = tHyperbolicParamList.sublist("Energy Conservation");
            if (tEnergyParamList.isSublist("Penalty Function"))
            {
                auto tPenaltyFuncList = tEnergyParamList.sublist("Penalty Function");
                mDiffusiveTermPenaltyExponent = tPenaltyFuncList.get<Plato::Scalar>("Diffusive Term Penalty Exponent", 3.0);
            }
        }
    }
};
// class InternalThermalForces

}
// namespace SIMP

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::InternalThermalForces, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::InternalThermalForces, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::InternalThermalForces, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::InternalThermalForces, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::InternalThermalForces, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::SIMP::InternalThermalForces, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
#endif
