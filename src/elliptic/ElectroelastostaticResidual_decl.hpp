#pragma once

#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ApplyWeighting.hpp"
#include "LinearElectroelasticMaterial.hpp"
#include "elliptic/AbstractVectorFunction.hpp"

#include "contact/AbstractSurfaceDisplacement.hpp"
#include "contact/AbstractContactForce.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElectroelastostaticResidual :
    public EvaluationType::ElementType,
    public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumDofsPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;

    static constexpr Plato::OrdinalType NElecDims = 1;
    static constexpr Plato::OrdinalType NMechDims = mNumSpatialDims;

    static constexpr Plato::OrdinalType EDofOffset = mNumSpatialDims;
    static constexpr Plato::OrdinalType MDofOffset = 0;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyEDispWeighting;
    ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms,  IndicatorFunctionType> mApplyStressWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;

    std::shared_ptr<Plato::NaturalBCs<ElementType, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<ElementType, NElecDims, mNumDofsPerNode, EDofOffset>> mBoundaryCharges;

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<mNumSpatialDims>> mMaterialModel;

    std::vector<std::string> mPlottable;

public:
    /**************************************************************************/
    ElectroelastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    );

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override;

    /**************************************************************************/
    void evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
    /**************************************************************************/

    /**************************************************************************/
    void evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
    /**************************************************************************/

    /******************************************************************************//**
     * \brief Evaluate contact
     *
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aSideSet side set to evaluate contact on
     * \param [in] aComputeSurfaceDisp functor for computing displacement on surface
     * \param [in] aComputeContactForce functor for computing contact force
     * \param [in] aState 2D array with state variables (C,DOF)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
    **********************************************************************************/
    void
    evaluate_contact(
        const Plato::SpatialModel                                                       & aSpatialModel,
        const std::string                                                               & aSideSet,
              Teuchos::RCP<Plato::Contact::AbstractSurfaceDisplacement<EvaluationType>>   aComputeSurfaceDisp,
              Teuchos::RCP<Plato::Contact::AbstractContactForce<EvaluationType>>          aComputeContactForce,
        const Plato::ScalarMultiVectorT <StateScalarType>                               & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType>                             & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>                              & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>                              & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override;
};
// class ElectroelastostaticResidual

} // namespace Elliptic

} // namespace Plato
