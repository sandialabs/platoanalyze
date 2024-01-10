/*
 * StructuralDynamicsResidual.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef STRUCTURALDYNAMICSRESIDUAL_HPP_
#define STRUCTURALDYNAMICSRESIDUAL_HPP_

#include <memory>

#include <Teuchos_ParameterList.hpp>

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "ElasticModelFactory.hpp"
#include "IsotropicLinearElasticMaterial.hpp"

#include "StateValues.hpp"
#include "ApplyPenalty.hpp"
#include "ComplexStrain.hpp"
#include "InertialForces.hpp"
#include "ApplyProjection.hpp"
#include "PlatoStaticsTypes.hpp"
#include "ComplexLinearStress.hpp"
#include "ComplexRayleighDamping.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "ComplexStressDivergence.hpp"
#include "SimplexStructuralDynamics.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "StructuralDynamicsCellResidual.hpp"

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType, class PenaltyFunctionType, class ProjectionType>
class StructuralDynamicsResidual:
        public Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>,
        public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    using Plato::Simplex<EvaluationType::SpatialDim>::mNumSpatialDims;
    using Plato::Simplex<EvaluationType::SpatialDim>::mNumNodesPerCell;

    using PhysicsType = typename Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>;

    using PhysicsType::mNumVoigtTerms;
    using PhysicsType::mComplexSpaceDim;
    using PhysicsType::mNumDofsPerCell;
    using PhysicsType::mNumDofsPerNode;

    using Plato::Elliptic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

private:
    Plato::Scalar mDensity;
    Plato::Scalar mMassPropDamp;
    Plato::Scalar mStiffPropDamp;

    ProjectionType mProjectionFunction;
    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyPenalty<PenaltyFunctionType> mApplyPenalty;
    Plato::ApplyProjection<ProjectionType> mApplyProjection;

    Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, PhysicsType>> mBodyLoads;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mBoundaryLoads;

public:
    /******************************************************************************//**
     *
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap problem-specific data storage
     * \param [in] aProblemParams parameter list with input data
     * \param [in] aPenaltyParams parameter list with penalty model input data
     *
    **********************************************************************************/
    explicit StructuralDynamicsResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
            FunctionBaseType(aSpatialDomain, aDataMap),
            mDensity(1.0),
            mMassPropDamp(0.0),
            mStiffPropDamp(0.0),
            mProjectionFunction(),
            mPenaltyFunction(aPenaltyParams),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mBodyLoads(nullptr),
            mCubatureRule(std::make_shared<CubatureType>()),
            mBoundaryLoads(nullptr)
    {
        this->initialize(aProblemParams);
    }

    /******************************************************************************//**
     *
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap problem-specific data storage
     * \param [in] aProblemParams parameter list with input data
     *
    **********************************************************************************/
    explicit StructuralDynamicsResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams
    ) :
            FunctionBaseType(aSpatialDomain, aDataMap),
            mDensity(1.0),
            mMassPropDamp(0.0),
            mStiffPropDamp(0.0),
            mProjectionFunction(),
            mPenaltyFunction(3.0, 0.0),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mBodyLoads(nullptr),
            mCubatureRule(std::make_shared<CubatureType>()),
            mBoundaryLoads(nullptr)
    {
        this->initialize(aProblemParams);
    }

    /******************************************************************************//**
     *
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap problem-specific data storage
     *
    **********************************************************************************/
    explicit StructuralDynamicsResidual(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
            FunctionBaseType(aSpatialDomain, aDataMap),
            mDensity(1.0),
            mMassPropDamp(0.0),
            mStiffPropDamp(0.0),
            mProjectionFunction(),
            mPenaltyFunction(3.0, 0.0),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mBodyLoads(nullptr),
            mCubatureRule(std::make_shared<CubatureType>()),
            mBoundaryLoads(nullptr)
    {
        this->initialize();
    }

    /******************************************************************************//**
     *
     * \brief Destructor
     *
    **********************************************************************************/
    ~StructuralDynamicsResidual()
    {
    }

    /******************************************************************************//**
     *
     * \brief Set mass proportional damping constant
     * \param [in] aInput mass proportional damping constant
     *
    **********************************************************************************/
    void setMassPropDamping(const Plato::Scalar& aInput)
    {
        mMassPropDamp = aInput;
    }

    /******************************************************************************//**
     *
     * \brief Set stiffness proportional damping constant
     * \param [in] aInput stiffness proportional damping constant
     *
    **********************************************************************************/
    void setStiffPropDamping(const Plato::Scalar& aInput)
    {
        mStiffPropDamp = aInput;
    }

    /******************************************************************************//**
     *
     * \brief Set material density
     * \param [in] aInput material density
     *
    **********************************************************************************/
    void setMaterialDensity(const Plato::Scalar& aInput)
    {
        mDensity = aInput;
    }

    /******************************************************************************//**
     *
     * \brief Set material stiffness constants (i.e. Lame constants)
     * \param [in] aInput material stiffness constants
     *
    **********************************************************************************/
    void setMaterialStiffnessConstants(const Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms>& aInput)
    {
        mCellStiffness = aInput;
    }

    /******************************************************************************//**
     *
     * \brief Set isotropic linear elastic material constants (i.e. Lame constants)
     * \param [in] aYoungsModulus Young's modulus
     * \param [in] aPoissonsRatio Poisson's ratio
     *
    **********************************************************************************/
    void setIsotropicLinearElasticMaterial(const Plato::Scalar& aYoungsModulus, const Plato::Scalar& aPoissonsRatio)
    {
        Plato::IsotropicLinearElasticMaterial<mNumSpatialDims> tMaterialModel(aYoungsModulus, aPoissonsRatio);
        mCellStiffness = tMaterialModel.getStiffnessMatrix();
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override 
    {
        return aSolutions;
    }

    /******************************************************************************//**
     *
     * \brief Evaluate structural dynamics residual.
     *
     * The structural dynamics residual is given by:
     *
     * \f$\mathbf{R} = \left(\mathbf{K} + \mathbf{C} - \omega^2\rho\mathbf{M}\right)
     * \mathbf{u}_{n} - \mathbf{f},
     *
     * where \f$n=1,\dots,N\f$ and \f$N\f$ is the number of frequency steps,
     * \f$\mathbf{M}$\f, \f$\mathbf{C}\f$ and \f$\mathbf{K}\f$ are the mass, damping
     * and stiffness matrices, \f$\mathbf{u}\f$ is the state vector, \f$\mathbf{f}\f$
     * is the force vector, \f$\omega\f$ is the angular frequency and \f$\rho\f$ is
     * the material density.
     *
     * \param [in] aState states per cells
     * \param [in] aControl controls per cells
     * \param [in] aConfiguration coordinates per cells
     * \param [in,out] aResidual residual per cells
     * \param [in] aAngularFrequency angular frequency
     *
    **********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfiguration,
                  Plato::ScalarMultiVectorT<ResultScalarType> & aResidual,
                  Plato::Scalar aAngularFrequency = 0.0) const
    {
        // Elastic force functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradientWorkset;
        Plato::ComplexStrain<mNumSpatialDims, mNumDofsPerNode> tComputeVoigtStrain;
        Plato::ComplexStressDivergence<mNumSpatialDims, mNumDofsPerNode> tComputeStressDivergence;
        Plato::ComplexLinearStress<mNumSpatialDims, mNumVoigtTerms> tComputeVoigtStress(mCellStiffness);
        // Damping force functors
        Plato::ComplexRayleighDamping<mNumSpatialDims> tComputeDamping(mMassPropDamp, mStiffPropDamp);
        // Inertial force functors
        Plato::StateValues tComputeStateValues;
        Plato::InertialForces tComputeInertialForces(mDensity);

        using StrainScalarType =
                typename Plato::fad_type_t<Plato::SimplexStructuralDynamics<mNumSpatialDims>, StateScalarType, ConfigScalarType>;
        // Elastic forces containers
        auto tNumCells = aState.extent(0);
        Plato::ScalarMultiVectorT<ResultScalarType> tElasticForces("ElasticForces", tNumCells, mNumDofsPerCell);
        Plato::ScalarArray3DT<StrainScalarType> tCellStrain("Strain", tNumCells, mComplexSpaceDim, mNumVoigtTerms);
        Plato::ScalarArray3DT<ResultScalarType> tCellStress("Stress", tNumCells, mComplexSpaceDim, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigScalarType> tCellGradient("Gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        // Damping forces containers
        Plato::ScalarMultiVectorT<ResultScalarType> tDampingForces("DampingForces", tNumCells, mNumDofsPerCell);
        // Inertial forces containers
        Plato::ScalarMultiVectorT<StateScalarType> tStateValues("StateValues", tNumCells, mNumDofsPerNode);
        Plato::ScalarMultiVectorT<ResultScalarType> tInertialForces("InertialForces", tNumCells, mNumDofsPerCell);
        // Cell volumes container
        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("Cell Volumes", tNumCells);

        // Copy data from host into device
        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;
        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        auto tOmegaTimesOmega = aAngularFrequency * aAngularFrequency;
        Kokkos::parallel_for("Structural Dynamics Residual Calculation", Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute elastic forces
            tComputeGradientWorkset(aCellOrdinal, tCellGradient, aConfiguration, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;
            tComputeVoigtStrain(aCellOrdinal, aState, tCellGradient, tCellStrain);
            tComputeVoigtStress(aCellOrdinal, tCellStrain, tCellStress);
            tComputeStressDivergence(aCellOrdinal, tCellVolume, tCellGradient, tCellStress, tElasticForces);
            // Compute inertial forces
            tComputeStateValues(aCellOrdinal, tBasisFunctions, aState, tStateValues);
            tComputeInertialForces(aCellOrdinal, tCellVolume, tBasisFunctions, tStateValues, tInertialForces);
            // Compute damping forces
            tComputeDamping(aCellOrdinal, tElasticForces, tInertialForces, tDampingForces);
            // Apply penalty to elastic, damping and inertial forces
            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl);
            tApplyPenalty(aCellOrdinal, tCellDensity, tDampingForces);
            tApplyPenalty(aCellOrdinal, tCellDensity, tElasticForces);
            tApplyPenalty(aCellOrdinal, tCellDensity, tInertialForces);
            // Compute structural dynamics residual
            Plato::structural_dynamics_cell_residual<mNumDofsPerCell>
                (aCellOrdinal, tOmegaTimesOmega, tElasticForces, tDampingForces, tInertialForces, aResidual);
        });

        // add body loads contribution
        if(mBodyLoads != nullptr)
        {
            auto tMesh = Plato::Elliptic::AbstractVectorFunction<EvaluationType>::getMesh();
            mBodyLoads->get(mSpatialDomain, aState, aControl, aResidual);
        }

    }

    /******************************************************************************//**
     *
     * \brief Evaluate structural dynamics boundary residual
     *
     * \param [in] aState states per cells
     * \param [in] aControl controls per cells
     * \param [in] aConfiguration coordinates per cells
     * \param [in,out] aResidual residual per cells
     * \param [in] aAngularFrequency angular frequency
     *
    **********************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfiguration,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResidual,
              Plato::Scalar aAngularFrequency = 0.0) const
    {
        // add neumann loads contribution
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfiguration, aResidual);
        }
    }

private:
    /**************************************************************************//**
     * 
     * \brief Checks if the material model sublist is defined in the input file.
     *
     * \param aParameterList Teuchos parameter list
     *
    ******************************************************************************/
    void isMaterialModelSublistDefined(Teuchos::ParameterList & aParameterList)
    {
        if(aParameterList.isSublist("Material Model") == false)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                << ", LINE: " << __LINE__ << "\nMESSAGE: MATERIAL MODEL SUBLIST IS NOT DEFINED IN THE INPUT FILE.\n"
                << "USER SHOULD DEFINE THE MATERIAL MODEL SUBLIST IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }
    
    /**************************************************************************//**
     * 
     * \brief Checks if the material density parameter is defined in the input file.
     *
     * \param aParameterList Teuchos parameter list
     *
    ******************************************************************************/
    void isMaterialDensityDefined(Teuchos::ParameterList & aParameterList)
    {
        auto tMaterialParamList = aParameterList.sublist("Material Model");
        if(tMaterialParamList.isParameter("Density") == false)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                << ", LINE: " << __LINE__ << "\nMESSAGE: MATERIAL DENSITY IS NOT DEFINED IN THE INPUT FILE.\n"
                << "USER SHOULD DEFINE THE MATERIAL DENSITY IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }
    
    /**************************************************************************//**
     * 
     * \brief Checks if the frequency steps sublist is defined in the input file.
     *
     * \param aParameterList Teuchos parameter list
     *
    ******************************************************************************/
    void isFrequencyStepsSublistDefined(Teuchos::ParameterList & aParameterList)
    {
        if(aParameterList.isSublist("Frequency Steps") == false)
        {
            std::ostringstream tErrorMessage;
            tErrorMessage << "\n\n************** ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__
                << ", LINE: " << __LINE__ << "\nMESSAGE: FREQUENCY STEPS SUBLIST IS NOT DEFINED IN THE INPUT FILE.\n"
                << "\nUSER SHOULD DEFINE THE FREQUENCY STEPS SUBLIST IN THE INPUT FILE. **************\n\n";
            throw std::runtime_error(tErrorMessage.str().c_str());
        }
    }

    /**************************************************************************//**
     *
     * \brief Initialize default material stiffness constants (i.e. Lame constants).
     *
    ******************************************************************************/
    void initialize()
    {
        // Create material model and get stiffness
        Teuchos::ParameterList tParamList;
        tParamList.set < Plato::Scalar > ("Poissons Ratio", 1.0);
        tParamList.set < Plato::Scalar > ("Youngs Modulus", 0.3);
        Plato::IsotropicLinearElasticMaterial<mNumSpatialDims> tDefaultMaterialModel(tParamList);
        mCellStiffness = tDefaultMaterialModel.getStiffnessMatrix();
    }

    /**************************************************************************//**
     *
     * \brief Initialize problem input parameters
     * \param [in] aParamList parameter list with input data
     *
    ******************************************************************************/
    void initialize(Teuchos::ParameterList & aParamList)
    {
        // Parse material density
        this->isMaterialModelSublistDefined(aParamList);
        this->isMaterialDensityDefined(aParamList);
        auto tMaterialParamList = aParamList.sublist("Material Model");
        mDensity = tMaterialParamList.get<Plato::Scalar>("Density");

        // Parse Rayleigh damping coefficients
        this->isFrequencyStepsSublistDefined(aParamList);
        mMassPropDamp = tMaterialParamList.get<Plato::Scalar>("Mass Proportional Damping", 0.0);
        mStiffPropDamp = tMaterialParamList.get<Plato::Scalar>("Stiffness Proportional Damping", 0.0);

        // Create material model and get stiffness
        Plato::ElasticModelFactory<mNumSpatialDims> tElasticModelFactory(aParamList);
        auto tMaterialModel = tElasticModelFactory.create();
        mCellStiffness = tMaterialModel->getStiffnessMatrix();

        // Parse body loads
        if(aParamList.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, PhysicsType>>(aParamList.sublist("Body Loads"));
        }

        // Parse Neumann loads
        if(aParamList.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(aParamList.sublist("Natural Boundary Conditions"));
        }
    }
};
// class StructuralDynamicsResidual

} // namespace Plato

#endif /* STRUCTURALDYNAMICSRESIDUAL_HPP_ */
