#pragma once

#include "Strain.hpp"
#include "ScalarGrad.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "ComputeDeviatoricStress.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "Simp.hpp"

#include "AbstractLocalVectorFunctionInc.hpp"
#include "ImplicitFunctors.hpp"
#include "AnalyzeMacros.hpp"

#include "J2PlasticityUtilities.hpp"
#include "ThermoPlasticityUtilities.hpp"

#include "ExpInstMacros.hpp"

//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
// NOTE: CURRENTLY THERE ARE TWO VERIONS OF J2PlasticityLocalResidual.hpp
//       THIS VERSION IS THE ORIGINAL VERSION. THE OTEHR VERSION
//       J2PlasticityLocalResidualExpFAD.hpp CONTAINS THE EXPRESSION FAD.
//
//       IF ANY CHANGES ARE MADE TO THIS FILE ONE MUST >>>>ALSO<<<<<
//       MAKE THE >>>>SAME<<<<< CHANGES TO J2PlasticityLocalResidualExpFAD.hpp
//
// This duplication is due to a risk aversion request. The decision
// was to have some minimal duplication for the time being so to avoid
// possible trouble down the road. There is another project outside
// Marie's and Sean's project that depends on the original
// implementation. The call to merge, i.e. delete the original
// implementation, will be made once a better understanding of how the
// expressionFad implementation will end-up long term.
//
//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

namespace Plato
{

/**************************************************************************//**
* \brief J2 Plasticity Local Residual class
******************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsType>
class J2PlasticityLocalResidual :
  public Plato::AbstractLocalVectorFunctionInc<EvaluationType>
{
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    static constexpr auto mNumDofsPerNode      = SimplexPhysicsType::mNumDofsPerNode;       /*!< number global degrees of freedom per node */
    static constexpr auto mNumNodesPerCell     = SimplexPhysicsType::mNumNodesPerCell;      /*!< number nodes per cell */
    static constexpr auto mNumStressTerms      = SimplexPhysicsType::mNumStressTerms;       /*!< number of stress/strain terms */
    static constexpr auto mNumLocalDofsPerCell = SimplexPhysicsType::mNumLocalDofsPerCell;  /*!< number of local degrees of freedom */

    using Plato::AbstractLocalVectorFunctionInc<EvaluationType>::mSpatialDomain;    /*!< Plato Analyze spatial domain */
    using Plato::AbstractLocalVectorFunctionInc<EvaluationType>::mDataMap; /*!< PLATO Engine output database */

    using GlobalStateT     = typename EvaluationType::StateScalarType;           /*!< global state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;       /*!< global state variables automatic differentiation type */
    using LocalStateT      = typename EvaluationType::LocalStateScalarType;      /*!< local state variables automatic differentiation type */
    using PrevLocalStateT  = typename EvaluationType::PrevLocalStateScalarType;  /*!< local state variables automatic differentiation type */
    using ControlT         = typename EvaluationType::ControlScalarType;         /*!< control variables automatic differentiation type */
    using ConfigT          = typename EvaluationType::ConfigScalarType;          /*!< config variables automatic differentiation type */
    using ResultT          = typename EvaluationType::ResultScalarType;          /*!< result variables automatic differentiation type */

    using CubatureType = Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>;

    Plato::Scalar mElasticShearModulus;            /*!< elastic shear modulus */

    Plato::Scalar mThermalExpansionCoefficient;    /*!< Thermal Expansivity */
    Plato::Scalar mReferenceTemperature;           /*!< reference temperature */
    Plato::Scalar mTemperatureScaling;             /*!< temperature scaling */

    Plato::Scalar mHardeningModulusIsotropic;      /*!< isotropic hardening modulus */
    Plato::Scalar mHardeningModulusKinematic;      /*!< kinematic hardening modulus */
    Plato::Scalar mInitialYieldStress;             /*!< initial yield stress */

    Plato::Scalar mElasticPropertiesPenaltySIMP;             /*!< SIMP penalty for elastic properties */
    Plato::Scalar mElasticPropertiesMinErsatzSIMP;           /*!< SIMP min ersatz stiffness for elastic properties */
    Plato::Scalar mAdditiveContinuationElasticProperties;    /*!< continuation parameter: multiplier on SIMP penalty for elastic properties */
    Plato::Scalar mUpperBoundOnElasticPropertiesPenaltySIMP; /*!< continuation parameter: upper bound on SIMP penalty for elastic properties */

    Plato::Scalar mPlasticPropertiesPenaltySIMP;             /*!< SIMP penalty for plastic properties */
    Plato::Scalar mPlasticPropertiesMinErsatzSIMP;           /*!< SIMP min ersatz stiffness for plastic properties */
    Plato::Scalar mAdditiveContinuationPlasticProperties;    /*!< continuation parameter: multiplier on SIMP penalty for plastic properties */
    Plato::Scalar mUpperBoundOnPlasticPropertiesPenaltySIMP; /*!< continuation parameter: upper bound on SIMP penalty for plastic properties */

    std::shared_ptr<CubatureType> mCubatureRule; /*!< linear tet cubature rule */

    const Plato::Scalar mSqrt3Over2 = std::sqrt(3.0/2.0);

    /**************************************************************************//**
    * \brief Return the names of the local state degrees of freedom
    * \return vector of local state dof names
    ******************************************************************************/
    std::vector<std::string> getLocalStateDofNames ()
    {
      if (mSpaceDim == 3)
      {
        std::vector<std::string> tDofNames(mNumLocalDofsPerCell);
        tDofNames[0]  = "Accumulated Plastic Strain";
        tDofNames[1]  = "Plastic Multiplier Increment";
        tDofNames[2]  = "Plastic Strain Tensor XX";
        tDofNames[3]  = "Plastic Strain Tensor YY";
        tDofNames[4]  = "Plastic Strain Tensor ZZ";
        tDofNames[5]  = "Plastic Strain Tensor YZ";
        tDofNames[6]  = "Plastic Strain Tensor XZ";
        tDofNames[7]  = "Plastic Strain Tensor XY";
        tDofNames[8]  = "Backstress Tensor XX";
        tDofNames[9]  = "Backstress Tensor YY";
        tDofNames[10] = "Backstress Tensor ZZ";
        tDofNames[11] = "Backstress Tensor YZ";
        tDofNames[12] = "Backstress Tensor XZ";
        tDofNames[13] = "Backstress Tensor XY";
        return tDofNames;
      }
      else if (mSpaceDim == 2)
      {
        std::vector<std::string> tDofNames(mNumLocalDofsPerCell);
        tDofNames[0] = "Accumulated Plastic Strain";
        tDofNames[1] = "Plastic Multiplier Increment";
        tDofNames[2] = "Plastic Strain Tensor XX";
        tDofNames[3] = "Plastic Strain Tensor YY";
        tDofNames[4] = "Plastic Strain Tensor XY";
        tDofNames[5] = "Plastic Strain Tensor ZZ";
        tDofNames[6] = "Backstress Tensor XX";
        tDofNames[7] = "Backstress Tensor YY";
        tDofNames[8] = "Backstress Tensor XY";
        tDofNames[9] = "Backstress Tensor ZZ";
        return tDofNames;
      }
      else
      {
        ANALYZE_THROWERR("J2 Plasticity Local Residual not implemented for space dim other than 2 or 3.")
      }
    }

    /**************************************************************************//**
    * \brief Initialize problem parameters
    * \param [in] aInputParams Teuchos parameter list
    ******************************************************************************/
    void initialize(Teuchos::ParameterList& aInputParams)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Teuchos::ParameterList tMaterialParamLists = aInputParams.sublist("Material Models");
        Teuchos::ParameterList tMaterialParamList  = tMaterialParamLists.sublist(tMaterialName);
        this->initializeIsotropicElasticMaterial(tMaterialParamList);
        this->initializeJ2Plasticity(tMaterialParamList);
    }

    /**************************************************************************//**
    * \brief Initialize isotropic material parameters
    * \param [in] aInputParams Teuchos parameter list
    ******************************************************************************/
    void initializeIsotropicElasticMaterial(Teuchos::ParameterList& aMaterialParams)
    {

        mTemperatureScaling = aMaterialParams.get<Plato::Scalar>("Temperature Scaling", 1.0);
        if( aMaterialParams.isSublist("Isotropic Linear Elastic") )
        {
          auto tElasticSubList = aMaterialParams.sublist("Isotropic Linear Elastic");
          mThermalExpansionCoefficient = 0.0;
          mReferenceTemperature        = 0.0;

          auto tElasticModulus = tElasticSubList.get<Plato::Scalar>("Youngs Modulus");
          auto tPoissonsRatio  = tElasticSubList.get<Plato::Scalar>("Poissons Ratio");
          mElasticShearModulus = tElasticModulus /
                  (static_cast<Plato::Scalar>(2.0) * (static_cast<Plato::Scalar>(1.0) + tPoissonsRatio));
        }
        else if( aMaterialParams.isSublist("Isotropic Linear Thermoelastic") )
        {
          auto tThermoelasticSubList = aMaterialParams.sublist("Isotropic Linear Thermoelastic");

          mThermalExpansionCoefficient = tThermoelasticSubList.get<Plato::Scalar>("Thermal Expansivity");
          mReferenceTemperature        = tThermoelasticSubList.get<Plato::Scalar>("Reference Temperature");

          auto tElasticModulus = tThermoelasticSubList.get<Plato::Scalar>("Youngs Modulus");
          auto tPoissonsRatio  = tThermoelasticSubList.get<Plato::Scalar>("Poissons Ratio");
          mElasticShearModulus = tElasticModulus /
                  (static_cast<Plato::Scalar>(2.0) * (static_cast<Plato::Scalar>(1.0) + tPoissonsRatio));
        }
        else
        {
          auto tMaterialName = mSpatialDomain.getMaterialName();
          std::stringstream ss;
          ss << "'Isotropic Linear Elastic' or 'Isotropic Linear Thermoelastic' sublist of '" << tMaterialName << "' does not exist.";
          ANALYZE_THROWERR(ss.str());
        }
    }

    /**************************************************************************//**
    * \brief Initialize J2 plasticity parameters
    * \param [in] aInputParams Teuchos parameter list
    ******************************************************************************/
    void initializeJ2Plasticity(Teuchos::ParameterList& aInputParams)
    {
        auto tPlasticityParamList = aInputParams.get<Teuchos::ParameterList>("Plasticity Model");
        if( tPlasticityParamList.isSublist("J2 Plasticity") )
        {
          auto tJ2PlasticitySubList = tPlasticityParamList.sublist("J2 Plasticity");
          this->checkJ2PlasticityInputs(tJ2PlasticitySubList);

          mHardeningModulusIsotropic = tJ2PlasticitySubList.get<Plato::Scalar>("Hardening Modulus Isotropic");
          mHardeningModulusKinematic = tJ2PlasticitySubList.get<Plato::Scalar>("Hardening Modulus Kinematic");
          mInitialYieldStress        = tJ2PlasticitySubList.get<Plato::Scalar>("Initial Yield Stress");

          mElasticPropertiesPenaltySIMP   = tJ2PlasticitySubList.get<Plato::Scalar>("Elastic Properties Penalty Exponent", 1.0);
          mElasticPropertiesMinErsatzSIMP = tJ2PlasticitySubList.get<Plato::Scalar>("Elastic Properties Minimum Ersatz", 1e-9);
          mAdditiveContinuationElasticProperties = tJ2PlasticitySubList.get<Plato::Scalar>("Elastic Properties Additive Continuation", 0.1);
          mUpperBoundOnElasticPropertiesPenaltySIMP = tJ2PlasticitySubList.get<Plato::Scalar>("Elastic Properties Penalty Exponent Upper Bound", 4.0);

          mPlasticPropertiesPenaltySIMP   = tJ2PlasticitySubList.get<Plato::Scalar>("Plastic Properties Penalty Exponent", 0.5);
          mPlasticPropertiesMinErsatzSIMP = tJ2PlasticitySubList.get<Plato::Scalar>("Plastic Properties Minimum Ersatz", 1e-4);
          mAdditiveContinuationPlasticProperties = tJ2PlasticitySubList.get<Plato::Scalar>("Plastic Properties Additive Continuation", 0.1);
          mUpperBoundOnPlasticPropertiesPenaltySIMP = tJ2PlasticitySubList.get<Plato::Scalar>("Plastic Properties Penalty Exponent Upper Bound", 3.5);
        }
        else
        {
            ANALYZE_THROWERR("'J2 Plasticity' sublist of 'Material Model' does not exist. Needed for J2Plasticity Implementation.")
        }
    }

    /**************************************************************************//**
    * \brief Check if all the required J2 plasticity parameters are defined.
    * \param [in] aInputParams Teuchos parameter list
    ******************************************************************************/
    void checkJ2PlasticityInputs(Teuchos::ParameterList& aInputParams)
    {
        const bool tRequriedInputParamsAreDefined = aInputParams.isParameter("Hardening Modulus Isotropic") &&
                aInputParams.isParameter("Hardening Modulus Kinematic") && aInputParams.isParameter("Initial Yield Stress");
        if(tRequriedInputParamsAreDefined == false)
        {
            std::string tError = std::string("Required input parameters, 'Hardening Modulus Isotropic', 'Hardening Modulus Kinematic', ") +
                    "and 'Initial Yield Stress', for J2 Plasticity model are not defined.";
            ANALYZE_THROWERR(tError)
        }
    }

    /**************************************************************************//**
    * \brief Update SIMP model used on the elastic properties.
    ******************************************************************************/
    void updateElasticPropertiesPenaltyModel()
    {
        auto tPreviousElasticPropertiesPenalty = mElasticPropertiesPenaltySIMP;
        auto tSuggestedElasticPropertiesPenalty = tPreviousElasticPropertiesPenalty + mAdditiveContinuationElasticProperties;
        mElasticPropertiesPenaltySIMP = tSuggestedElasticPropertiesPenalty >= mUpperBoundOnElasticPropertiesPenaltySIMP ?
                mUpperBoundOnElasticPropertiesPenaltySIMP : tSuggestedElasticPropertiesPenalty;

        std::ostringstream tMsg;
        tMsg << "J2 Plasticity Local Residual: New penalty parameter for the elastic properties is set to '"
             << mElasticPropertiesPenaltySIMP << "'. Previous penalty parameter was '" << tPreviousElasticPropertiesPenalty
             << "'.\n";
        REPORT(tMsg.str().c_str())
    }

    /**************************************************************************//**
    * \brief Update SIMP model used on the plastic properties.
    ******************************************************************************/
    void updatePlasticPropertiesPenaltyModel()
    {
        auto tPreviousPlasticPropertiesPenalty = mPlasticPropertiesPenaltySIMP;
        auto tSuggestedPlasticPropertiesPenalty = tPreviousPlasticPropertiesPenalty + mAdditiveContinuationPlasticProperties;
        mPlasticPropertiesPenaltySIMP = tSuggestedPlasticPropertiesPenalty >= mUpperBoundOnPlasticPropertiesPenaltySIMP ?
                mUpperBoundOnPlasticPropertiesPenaltySIMP : tSuggestedPlasticPropertiesPenalty;

        std::ostringstream tMsg;
        tMsg << "J2 Plasticity Local Residual: New penalty parameter for the plastic properties is set to '"
             << mPlasticPropertiesPenaltySIMP << "'. Previous penalty parameter was '" << tPreviousPlasticPropertiesPenalty
             << "'.\n";
        REPORT(tMsg.str().c_str())
    }

public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialDomain Plato Analyze spatial domain
    * \param [in] aDataMap problem-specific data map
    * \param [in] aProblemParams Teuchos parameter list
    ******************************************************************************/
    J2PlasticityLocalResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams
    ) :
        AbstractLocalVectorFunctionInc<EvaluationType>(aSpatialDomain, aDataMap, getLocalStateDofNames() ),
        mCubatureRule(std::make_shared<CubatureType>())
    {
        this->initialize(aProblemParams);
    }

    /**************************************************************************//**
    * \brief Evaluate the local J2 plasticity residual
    * \param [in] aGlobalState global state at current time step
    * \param [in] aPrevGlobalState global state at previous time step
    * \param [in] aLocalState local state at current time step
    * \param [in] aPrevLocalState local state at previous time step
    * \param [in] aControl control parameters
    * \param [in] aConfig configuration parameters
    * \param [out] aResult evaluated local residuals
    ******************************************************************************/
    virtual void
    evaluate(
        const Plato::ScalarMultiVectorT< GlobalStateT >     & aGlobalState,
        const Plato::ScalarMultiVectorT< PrevGlobalStateT > & aPrevGlobalState,
        const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
        const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
        const Plato::ScalarMultiVectorT< ControlT >         & aControl,
        const Plato::ScalarArray3DT    < ConfigT >          & aConfig,
        const Plato::ScalarMultiVectorT< ResultT >          & aResult,
        const Plato::TimeData                               & aTimeData
    ) const
    {
      auto tNumCells = mSpatialDomain.numCells();

      using TotalStrainT   = typename Plato::fad_type_t<SimplexPhysicsType, GlobalStateT, ConfigT>;
      using ElasticStrainT = typename Plato::fad_type_t<SimplexPhysicsType, LocalStateT, ConfigT, GlobalStateT>;
      using StressT        = typename Plato::fad_type_t<SimplexPhysicsType, ControlT, LocalStateT, ConfigT, GlobalStateT>;

      // Functors
      Plato::ComputeGradientWorkset<mSpaceDim>  tComputeGradient;
      Plato::Strain<mSpaceDim, mNumDofsPerNode> tComputeTotalStrain;
      Plato::ComputeDeviatoricStress<mSpaceDim> tComputeDeviatoricStress;

      // J2 Utility Functions Object
      Plato::J2PlasticityUtilities<mSpaceDim>   tJ2PlasticityUtils;

      // ThermoPlasticity Utility Functions Object (for computing elastic strain and potentially temperature-dependent material properties)
      Plato::ThermoPlasticityUtilities<mSpaceDim, SimplexPhysicsType>
            tThermoPlasticityUtils(mThermalExpansionCoefficient, mReferenceTemperature, mTemperatureScaling);

      // Many views
      Plato::ScalarVectorT<ConfigT>             tCellVolume("cell volume unused", tNumCells);
      Plato::ScalarVectorT<StressT>             tDevStressMinusBackstressNorm("norm(deviatoric_stress - backstress)",tNumCells);
      Plato::ScalarArray3DT<ConfigT>            tGradient("gradient", tNumCells,mNumNodesPerCell,mSpaceDim);
      Plato::ScalarMultiVectorT<StressT>        tDeviatoricStress("deviatoric stress", tNumCells,mNumStressTerms);
      Plato::ScalarMultiVectorT<StressT>        tYieldSurfaceNormal("yield surface normal",tNumCells,mNumStressTerms);
      Plato::ScalarMultiVectorT<TotalStrainT>   tTotalStrain("total strain",tNumCells,mNumStressTerms);
      Plato::ScalarMultiVectorT<ElasticStrainT> tElasticStrain("elastic strain", tNumCells,mNumStressTerms);

      // Transfer elasticity parameters to device
      auto tElasticShearModulus = mElasticShearModulus;

      // Transfer plasticity parameters to device
      auto tHardeningModulusIsotropic = mHardeningModulusIsotropic;
      auto tHardeningModulusKinematic = mHardeningModulusKinematic;
      auto tInitialYieldStress        = mInitialYieldStress;

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tSqrt3Over2 = mSqrt3Over2;

      Plato::MSIMP tElasticPropertiesSIMP(mElasticPropertiesPenaltySIMP, mElasticPropertiesMinErsatzSIMP);
      Plato::MSIMP tPlasticPropertiesSIMP(mPlasticPropertiesPenaltySIMP, mPlasticPropertiesMinErsatzSIMP);

      Kokkos::parallel_for("Compute cell local residuals", Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);

        // compute elastic strain
        tComputeTotalStrain(aCellOrdinal, tTotalStrain, aGlobalState, tGradient);
        tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aGlobalState, aLocalState,
                                                    tBasisFunctions, tTotalStrain, tElasticStrain);

        // apply penalization to elastic shear modulus
        ControlT tDensity               = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControl);
        ControlT tElasticParamsPenalty  = tElasticPropertiesSIMP(tDensity);
        ControlT tPenalizedShearModulus = tElasticParamsPenalty * tElasticShearModulus;

        // compute deviatoric stress
        tComputeDeviatoricStress(aCellOrdinal, tPenalizedShearModulus, tElasticStrain, tDeviatoricStress);

        // compute eta = (deviatoric_stress - backstress) ... and its norm ... the normalized version is the yield surface normal
        tJ2PlasticityUtils.computeDeviatoricStressMinusBackstressNormalized(aCellOrdinal, tDeviatoricStress, aLocalState,
                                                                            tYieldSurfaceNormal, tDevStressMinusBackstressNorm);

        // apply penalization to plasticity material parameters
        ControlT tPlasticParamsPenalty               = tPlasticPropertiesSIMP(tDensity);
        ControlT tPenalizedHardeningModulusIsotropic = tPlasticParamsPenalty * tHardeningModulusIsotropic;
        ControlT tPenalizedHardeningModulusKinematic = tPlasticParamsPenalty * tHardeningModulusKinematic;
        ControlT tPenalizedInitialYieldStress        = tPlasticParamsPenalty * tInitialYieldStress;

        // compute yield stress
        ResultT tYieldStress = tPenalizedInitialYieldStress +
                               tPenalizedHardeningModulusIsotropic * aLocalState(aCellOrdinal, 0); // SHOULD THIS BE PREV? I think no.

        // ### ELASTIC STEP ###
        // Residual: Accumulated Plastic Strain, DOF: Accumulated Plastic Strain
        aResult(aCellOrdinal, 0) = aLocalState(aCellOrdinal, 0) - aPrevLocalState(aCellOrdinal, 0);

        // Residual: Plastic Multiplier Increment = 0 , DOF: Plastic Multiplier Increment
        aResult(aCellOrdinal, 1) = aLocalState(aCellOrdinal, 1);

        // Residual: Plastic Strain Tensor, DOF: Plastic Strain Tensor
        tJ2PlasticityUtils.fillPlasticStrainTensorResidualElasticStep(aCellOrdinal, aLocalState, aPrevLocalState, aResult);

        // Residual: Backstress, DOF: Backstress
        tJ2PlasticityUtils.fillBackstressTensorResidualElasticStep(aCellOrdinal, aLocalState, aPrevLocalState, aResult);

        if (aLocalState(aCellOrdinal, 1) /*Current Plastic Multiplier Increment*/ > 0.0) // -> yielding (assumes local state already updated)
        {
          // Residual: Accumulated Plastic Strain, DOF: Accumulated Plastic Strain
          aResult(aCellOrdinal, 0) = aLocalState(aCellOrdinal, 0) - aPrevLocalState(aCellOrdinal, 0)
                                                                  -     aLocalState(aCellOrdinal, 1);

          // Residual: Yield Function , DOF: Plastic Multiplier Increment
          aResult(aCellOrdinal, 1) = tSqrt3Over2 * tDevStressMinusBackstressNorm(aCellOrdinal) - tYieldStress;

          // Residual: Plastic Strain Tensor, DOF: Plastic Strain Tensor
          tJ2PlasticityUtils.fillPlasticStrainTensorResidualPlasticStep(aCellOrdinal, aLocalState, aPrevLocalState,
                                                                        tYieldSurfaceNormal, aResult);

          // Residual: Backstress, DOF: Backstress
          tJ2PlasticityUtils.fillBackstressTensorResidualPlasticStep(aCellOrdinal,
                                                                     tPenalizedHardeningModulusKinematic,
                                                                     aLocalState,         aPrevLocalState,
                                                                     tYieldSurfaceNormal, aResult);
        }

      });
    }

    /**************************************************************************//**
    * \brief Update the local state variables
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [out] aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aConfig configuration parameters
    ******************************************************************************/
    virtual void
    updateLocalState(
        const Plato::ScalarMultiVector & aGlobalState,
        const Plato::ScalarMultiVector & aPrevGlobalState,
        const Plato::ScalarMultiVector & aLocalState,
        const Plato::ScalarMultiVector & aPrevLocalState,
        const Plato::ScalarMultiVector & aControl,
        const Plato::ScalarArray3D     & aConfig,
        const Plato::TimeData          & aTimeData
    ) const
    {
      auto tNumCells = mSpatialDomain.numCells();

      // Functors
      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::Strain<mSpaceDim, mNumDofsPerNode> tComputeTotalStrain;
      Plato::ComputeDeviatoricStress<mSpaceDim> tComputeDeviatoricStress;

      // J2 Utility Functions Object
      Plato::J2PlasticityUtilities<mSpaceDim>  tJ2PlasticityUtils;

      // ThermoPlasticity Utility Functions Object (for computing elastic strain and potentially temperature-dependent material properties)
      Plato::ThermoPlasticityUtilities<mSpaceDim, SimplexPhysicsType>
            tThermoPlasticityUtils(mThermalExpansionCoefficient, mReferenceTemperature, mTemperatureScaling);

      // Many views
      Plato::ScalarVector      tCellVolume("cell volume unused",tNumCells);
      Plato::ScalarMultiVector tTotalStrain("total strain",tNumCells,mNumStressTerms);
      Plato::ScalarMultiVector tElasticStrain("elastic strain",tNumCells,mNumStressTerms);
      Plato::ScalarArray3D     tGradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);
      Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress",tNumCells,mNumStressTerms);
      Plato::ScalarMultiVector tYieldSurfaceNormal("yield surface normal",tNumCells,mNumStressTerms);
      Plato::ScalarVector      tDevStressMinusBackstressNorm("||(deviatoric stress - backstress)||",tNumCells);

      // Transfer elasticity parameters to device
      auto tElasticShearModulus = mElasticShearModulus;

      // Transfer plasticity parameters to device
      auto tHardeningModulusIsotropic = mHardeningModulusIsotropic;
      auto tHardeningModulusKinematic = mHardeningModulusKinematic;
      auto tInitialYieldStress        = mInitialYieldStress;

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tSqrt3Over2 = mSqrt3Over2;

      Plato::MSIMP tElasticPropertiesSIMP(mElasticPropertiesPenaltySIMP, mElasticPropertiesMinErsatzSIMP);
      Plato::MSIMP tPlasticPropertiesSIMP(mPlasticPropertiesPenaltySIMP, mPlasticPropertiesMinErsatzSIMP);

      Kokkos::parallel_for("Update local state dofs", Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);

        // Accumulated Plastic Strain
        aLocalState(aCellOrdinal, 0) = aPrevLocalState(aCellOrdinal, 0);

        // Plastic Multiplier Increment
        aLocalState(aCellOrdinal, 1) = 0.0;

        tJ2PlasticityUtils.updatePlasticStrainAndBackstressElasticStep(aCellOrdinal, aPrevLocalState, aLocalState);

        // compute elastic strain
        tComputeTotalStrain(aCellOrdinal, tTotalStrain, aGlobalState, tGradient);
        tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aGlobalState, aLocalState,
                                                    tBasisFunctions, tTotalStrain, tElasticStrain);

        // apply penalization to elastic shear modulus
        Plato::Scalar tDensity               = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControl);
        Plato::Scalar tElasticParamsPenalty  = tElasticPropertiesSIMP(tDensity);
        Plato::Scalar tPenalizedShearModulus = tElasticParamsPenalty * tElasticShearModulus;

        // compute deviatoric stress
        tComputeDeviatoricStress(aCellOrdinal, tPenalizedShearModulus, tElasticStrain, tDeviatoricStress);

        // compute eta = (deviatoric_stress - backstress) ... and its norm ... the normalized version is the yield surf normal
        tJ2PlasticityUtils.computeDeviatoricStressMinusBackstressNormalized(aCellOrdinal, tDeviatoricStress, aLocalState,
                                                                            tYieldSurfaceNormal, tDevStressMinusBackstressNorm);

        // apply penalization to plasticity material parameters
        Plato::Scalar tPlasticParamsPenalty               = tPlasticPropertiesSIMP(tDensity);
        Plato::Scalar tPenalizedHardeningModulusIsotropic = tPlasticParamsPenalty * tHardeningModulusIsotropic;
        Plato::Scalar tPenalizedHardeningModulusKinematic = tPlasticParamsPenalty * tHardeningModulusKinematic;
        Plato::Scalar tPenalizedInitialYieldStress        = tPlasticParamsPenalty * tInitialYieldStress;

        // compute yield stress
        Plato::Scalar tYieldStress = tPenalizedInitialYieldStress +
                                     tPenalizedHardeningModulusIsotropic * aLocalState(aCellOrdinal, 0);

        // compute the yield function at the trial state
        Plato::Scalar tTrialStateYieldFunction = tSqrt3Over2 * tDevStressMinusBackstressNorm(aCellOrdinal) - tYieldStress;

        if (tTrialStateYieldFunction > static_cast<Plato::Scalar>(1.0e-10)) // plastic step
        {
          // Plastic Multiplier Increment (for J2 w/ linear isotropic/kinematic hardening -> analytical return mapping)
          aLocalState(aCellOrdinal, 1) = tTrialStateYieldFunction / (static_cast<Plato::Scalar>(3.0) * tPenalizedShearModulus +
                                         tPenalizedHardeningModulusIsotropic + tPenalizedHardeningModulusKinematic);

          // Accumulated Plastic Strain
          aLocalState(aCellOrdinal, 0) = aPrevLocalState(aCellOrdinal, 0) + aLocalState(aCellOrdinal, 1);

          tJ2PlasticityUtils.updatePlasticStrainAndBackstressPlasticStep(aCellOrdinal, aPrevLocalState, tYieldSurfaceNormal,
                                                                         tPenalizedHardeningModulusKinematic, aLocalState);
        }
      });
    }

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aGlobalState,
                       const Plato::ScalarMultiVector & aLocalState,
                       const Plato::ScalarVector & aControl,
                       const Plato::TimeData     & aTimeData) override
    {
        this->updateElasticPropertiesPenaltyModel();
        this->updatePlasticPropertiesPenaltyModel();
    }
};
// class J2PlasticityLocalResidual

}
// namespace Plato

#include "SimplexPlasticity.hpp"
#include "SimplexThermoPlasticity.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEC_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 2)
#endif
#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEC_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 3)
#endif
