/*
 * BrinkmanForces.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "UtilsTeuchos.hpp"
#include "SpatialModel.hpp"
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/fluids/FluidsUtils.hpp"
#include "hyperbolic/fluids/SimplexFluids.hpp"
#include "hyperbolic/fluids/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/fluids/MomentumConservationUtils.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam PhysicsT    Physics Type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) Evaluation Type
 *
 * \class BrinkmanForces
 *
 * \brief Class responsible for the evaluation of the Brinkman forces (i.e.
 *   fictitious material model), including the stabilization forces associated
 *   with the Brinkman forces.
 *
 * Thermal Buoyancy Force:
 *   \[ \Delta{t}\gamma \int_{\Omega_e} w_i u_i^n d\Omega_e \]
 *
 * Stabilized Buoyancy Force:
 *   \[ \frac{\Delta{t}^2}{2}*\gamma \int_{\Omega_e} (\frac{\partial w_i}{\partial x_k} u_k^n ) u_i^n d\Omega_e \]
 *
 * where \f$ \gamma \f$ is the impermeability constant, \f$ \Delta{t} \f$ is the
 * current time step, \f$ u_i^n \f$ is the i-th component of the previous velocity
 * field and \f$ x_i \f$ is the i-th coordinate.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class BrinkmanForces
{
private:
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */

    // set local ad types
    using ResultT  = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT  = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using ControlT = typename EvaluationT::ControlScalarType; /*!< control FAD type */
    using PrevVelT = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD type */

    Plato::Scalar mStabilization = 1.0; /*!< stabilization multiplier */
    Plato::Scalar mImpermeability = 1.0; /*!< permeability dimensionless number */
    Plato::Scalar mBrinkmanConvexityParam = 0.5;  /*!< brinkman model convexity parameter */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output data map
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    BrinkmanForces
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setImpermeability(aInputs);
        mStabilization = Plato::Fluids::stabilization_constant("Momentum Conservation", aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    ~BrinkmanForces(){}

    /***************************************************************************//**
     * \brief Evaluate Brinkman forces, including stabilized forces if enabled.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS) const
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            ANALYZE_THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set temporary local worksets
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarMultiVectorT<ResultT> tBrinkman("cell brinkman forces", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("cell previous velocity", tNumCells, mNumSpatialDims);

        // set input worksets
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member host scalar data to device
        auto tStabilization = mStabilization;
        auto tImpermeability = mImpermeability;
        auto tBrinkmanConvexityParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for("brinkman force evaluator", Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add brinkman force contribution to residual, R += \Delta{t}\gamma M u^n
            auto tMultiplier = static_cast<Plato::Scalar>(1.0) * tCriticalTimeStep(0);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            ControlT tPenalizedPermeability = Plato::Fluids::brinkman_penalization<mNumNodesPerCell>
                (aCellOrdinal, tImpermeability, tBrinkmanConvexityParam, tControlWS);
            Plato::Fluids::calculate_brinkman_forces<mNumSpatialDims>
                (aCellOrdinal, tPenalizedPermeability, tPrevVelGP, tBrinkman);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tBrinkman, aResultWS, tMultiplier);

            // 2. add stabilizing brinkman force to residual, R += (\frac{\Delta{t}^2}{2}\gamma) M_u u_n
            tMultiplier = tStabilization * static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_vector_force<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tBrinkman, aResultWS, tMultiplier);
        });
    }

private:
    /***************************************************************************//**
     * \brief Set impermeability constant.
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    void setImpermeability
    (Teuchos::ParameterList & aInputs)
    {
        auto tMyMaterialName = mSpatialDomain.getMaterialName();
        if( Plato::Fluids::is_material_property_defined("Impermeability Number", tMyMaterialName, aInputs) )
        {
            // all use cases
            mImpermeability = Plato::Fluids::get_material_property<Plato::Scalar>("Impermeability Number", tMyMaterialName, aInputs);
        }
        else if( Plato::Fluids::is_material_property_defined("Darcy Number", tMyMaterialName, aInputs) && 
            Plato::Fluids::is_material_property_defined("Prandtl Number", tMyMaterialName, aInputs) )
        {
            // natural buoyancy 
            auto tDaNum = Plato::Fluids::get_material_property<Plato::Scalar>("Darcy Number", tMyMaterialName, aInputs);
            auto tPrNum = Plato::Fluids::get_material_property<Plato::Scalar>("Prandtl Number", tMyMaterialName, aInputs);
            mImpermeability = tPrNum / tDaNum;
        }
        else if( Plato::Fluids::is_material_property_defined("Darcy Number", tMyMaterialName, aInputs) && 
            Plato::Fluids::is_material_property_defined("Reynolds Number", tMyMaterialName, aInputs) )
        {
            // forced convection
            auto tDaNum = Plato::Fluids::get_material_property<Plato::Scalar>("Darcy Number", tMyMaterialName, aInputs);
            auto tReNum = Plato::Fluids::get_material_property<Plato::Scalar>("Reynolds Number", tMyMaterialName, aInputs);
            mImpermeability = static_cast<Plato::Scalar>(1.0) / (tDaNum * tReNum);
        }
        else
        {
            // default value for all cases
            mImpermeability = 1e2;
        }
    }
};
// class BrinkmanForces

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::BrinkmanForces, Plato::MomentumConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::BrinkmanForces, Plato::MomentumConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::BrinkmanForces, Plato::MomentumConservation, Plato::SimplexFluids, 3, 1)
#endif
