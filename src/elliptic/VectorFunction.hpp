#pragma once

#include <memory>

#include "WorksetBase.hpp"
#include "ImplicitFunctors.hpp"
#include "MatrixGraphUtils.hpp"
#include "NaturalBCs.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/EvaluationTypes.hpp"

#include "contact/SurfaceDisplacementFactory.hpp"
#include "contact/ContactForceFactory.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   and manages the evaluation of the function and derivatives wrt state
   and control.
  
*/
/******************************************************************************/
template<typename PhysicsType>
class VectorFunction : public Plato::WorksetBase<typename PhysicsType::ElementType>
{
  private:

    using ElementType = typename PhysicsType::ElementType;

    using Plato::WorksetBase<ElementType>::mNumDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell;
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::WorksetBase<ElementType>::mNumControl;
    using Plato::WorksetBase<ElementType>::mNumNodes;
    using Plato::WorksetBase<ElementType>::mNumCells;

    template<typename EvaluationType>
    using EvaluationFunction = std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>;

    template<typename EvaluationType>
    using EvaluationFunctionMap = std::map<std::string, EvaluationFunction<EvaluationType>>;

    using Residual  = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using Jacobian  = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using GradientX = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

    using ResidualFunction  = EvaluationFunction<Residual>;
    using JacobianFunction  = EvaluationFunction<Jacobian>;
    using GradientXFunction = EvaluationFunction<GradientX>;
    using GradientZFunction = EvaluationFunction<GradientZ>;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    EvaluationFunctionMap<Residual>  mResidualFunctions;
    EvaluationFunctionMap<Jacobian>  mJacobianFunctions;
    EvaluationFunctionMap<GradientX> mGradientXFunctions;
    EvaluationFunctionMap<GradientZ> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap      & mDataMap;

  public:

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map
    * \param [in] aProblemParams Teuchos parameter list with input data
    * \param [in] aProblemType problem type
    *
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aProblemType
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
    {
        typename PhysicsType::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
          auto tName = tDomain.getDomainName();
          mResidualFunctions [tName] = tFunctionFactory.template createVectorFunction<Residual> (tDomain, aDataMap, aProblemParams, aProblemType);
          mJacobianFunctions [tName] = tFunctionFactory.template createVectorFunction<Jacobian> (tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientZFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientZ>(tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientXFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientX>(tDomain, aDataMap, aProblemParams, aProblemType);
        }
    }

    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
    {
    }

    /**************************************************************************//**
    * \brief Return dof names for Physics
    * \return dof names
    ******************************************************************************/
    std::vector<std::string> getDofNames() const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResidualFunctions.at(tFirstBlockName)->getDofNames();
    }

    /**************************************************************************//**
    * \brief Return number of nodes on the mesh
    * \return number of nodes
    ******************************************************************************/
    Plato::OrdinalType numNodes() const
    {
        return (mNumNodes);
    }

    /**************************************************************************//**
    * \brief Return number of elements/cells on the mesh
    * \return number of elements
    ******************************************************************************/
    Plato::OrdinalType numCells() const
    {
        return (mNumCells);
    }

    /**************************************************************************//**
    * \brief Return total number of global degrees of freedom
    * \return total number of global degrees of freedom
    ******************************************************************************/
    Plato::OrdinalType numDofsPerCell() const
    {
        return (mNumDofsPerCell);
    }

    /**************************************************************************//**
    * \brief Return total number of nodes per cell/element
    * \return total number of nodes per cell/element
    ******************************************************************************/
    Plato::OrdinalType numNodesPerCell() const
    {
        return (mNumNodesPerCell);
    }

    /**************************************************************************//**
    * \brief Return number of degrees of freedom per node
    * \return number of degrees of freedom per node
    ******************************************************************************/
    Plato::OrdinalType numDofsPerNode() const
    {
        return (mNumDofsPerNode);
    }

    /**************************************************************************//**
    * \brief Return number of control vectors/fields, e.g. number of materials.
    * \return number of control vectors
    ******************************************************************************/
    Plato::OrdinalType numControlsPerNode() const
    {
        return (mNumControl);
    }

    /**************************************************************************//**
    *
    * \brief Allocate residual evaluator
    * \param [in] aResidual residual evaluator
    * \param [in] aJacobian Jacobian evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const ResidualFunction & aResidual,
        const JacobianFunction & aJacobian,
              std::string        aName
    )
    {
        mResidualFunctions[aName] = aResidual;
        mJacobianFunctions[aName] = aJacobian;
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to control evaluator
    * \param [in] aGradientZ partial derivative with respect to control evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const GradientZFunction & aGradientZ,
              std::string         aName
    )
    {
        mGradientZFunctions[aName] = aGradientZ; 
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to configuration evaluator
    * \param [in] GradientX partial derivative with respect to configuration evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const GradientXFunction & aGradientX,
              std::string         aName
    )
    {
        mGradientXFunctions[aName] = aGradientX; 
    }

    /**************************************************************************//**
    *
    * \brief Return local number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
    }

    /****************************************************************************//**
    * \brief Function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        auto tItr = mResidualFunctions.find(tFirstBlockName);
        if(tItr == mResidualFunctions.end())
            { ANALYZE_THROWERR(std::string("Element block with name '") + tFirstBlockName + "is not defined in residual function to element block map.") }
        return tItr->second->getSolutionStateOutputData(aSolutions);
    }

    template<typename EvaluationType>
    Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> 
    internalForceContribution
    (const EvaluationFunction<EvaluationType> & aFunction,
     const Plato::SpatialDomain               & aDomain,
     const Plato::ScalarVector                & aState,
     const Plato::ScalarVector                & aControl,
           Plato::Scalar                        aTimeStep = 0.0) const
    {
        using ConfigScalar  = typename EvaluationType::ConfigScalarType;
        using StateScalar   = typename EvaluationType::StateScalarType;
        using ControlScalar = typename EvaluationType::ControlScalarType;
        using ResultScalar  = typename EvaluationType::ResultScalarType;

        auto tNumCells = aDomain.numCells();

        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, aDomain);

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
        Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, aDomain);

        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
        Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, aDomain);

        Plato::ScalarMultiVectorT<ResultScalar> tValues("Values", tNumCells, mNumDofsPerCell);

        aFunction->evaluate( tStateWS, tControlWS, tConfigWS, tValues, aTimeStep );

        return tValues;
    }

    template<typename EvaluationType>
    Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> 
    externalForceContribution
    (const EvaluationFunction<EvaluationType> & aFunction,
     const Plato::ScalarVector                & aState,
     const Plato::ScalarVector                & aControl,
           Plato::Scalar                        aTimeStep = 0.0) const
    {
        using ConfigScalar  = typename EvaluationType::ConfigScalarType;
        using StateScalar   = typename EvaluationType::StateScalarType;
        using ControlScalar = typename EvaluationType::ControlScalarType;
        using ResultScalar  = typename EvaluationType::ResultScalarType;

        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

        Plato::ScalarMultiVectorT<ResultScalar> tValues("Values", mNumCells, mNumDofsPerCell);

        aFunction->evaluate_boundary(mSpatialModel, tStateWS, tControlWS, tConfigWS, tValues, aTimeStep );

        return tValues;
    }

    template<typename EvaluationType>
    Plato::ScalarMultiVectorT<typename EvaluationType::ResultScalarType> 
    contactForceContribution
    (const EvaluationFunction<EvaluationType> & aFunction,
     const Plato::ScalarVector                & aState,
     const Plato::ScalarVector                & aControl,
           Plato::Scalar                        aTimeStep = 0.0) const
    {
        using ConfigScalar  = typename EvaluationType::ConfigScalarType;
        using StateScalar   = typename EvaluationType::StateScalarType;
        using ControlScalar = typename EvaluationType::ControlScalarType;
        using ResultScalar  = typename EvaluationType::ResultScalarType;

        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

        auto tPairs = mSpatialModel.contactPairs();
        Plato::Contact::SurfaceDisplacementFactory<EvaluationType> tSurfaceDispFactory;
        Plato::Contact::ContactForceFactory<EvaluationType>        tContactForceFactory;

        Plato::ScalarMultiVectorT<ResultScalar> tValues("", mNumCells, mNumDofsPerCell);

        for (auto tPair : tPairs)
        {
            auto computeContactForce = tContactForceFactory.create(tPair.penaltyType, tPair.penaltyValue);

            auto tSideSet = tPair.surfaceA.childSideSet();
            auto computeChildSurfaceDispA = tSurfaceDispFactory.createChildContribution(tPair.surfaceA);
            aFunction->evaluate_contact(mSpatialModel, tSideSet, computeChildSurfaceDispA, computeContactForce, tStateWS, tControlWS, tConfigWS, tValues, aTimeStep);

            auto computeParentSurfaceDispA = tSurfaceDispFactory.createParentContribution(tPair.surfaceA, mSpatialModel.Mesh, -1.0);
            for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
            {
                computeParentSurfaceDispA->setChildNode(iChildNode);
                aFunction->evaluate_contact(mSpatialModel, tSideSet, computeParentSurfaceDispA, computeContactForce, tStateWS, tControlWS, tConfigWS, tValues, aTimeStep);
            }

            tSideSet = tPair.surfaceB.childSideSet();
            auto computeChildSurfaceDispB = tSurfaceDispFactory.createChildContribution(tPair.surfaceB);
            aFunction->evaluate_contact(mSpatialModel, tSideSet, computeChildSurfaceDispB, computeContactForce, tStateWS, tControlWS, tConfigWS, tValues, aTimeStep);

            auto computeParentSurfaceDispB = tSurfaceDispFactory.createParentContribution(tPair.surfaceB, mSpatialModel.Mesh, -1.0);
            for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
            {
                computeParentSurfaceDispB->setChildNode(iChildNode);
                aFunction->evaluate_contact(mSpatialModel, tSideSet, computeParentSurfaceDispB, computeContactForce, tStateWS, tControlWS, tConfigWS, tValues, aTimeStep);
            }
        }

        return tValues;
    }

    template<typename EvaluationType, typename EntryOrdinalType>
    void contactForceNonlocalGradient
    (const EvaluationFunction<EvaluationType> & aFunction,
           Teuchos::RCP<Plato::CrsMatrixType>   aInputMatrix,
     const EntryOrdinalType                   & aEntryOrdinal,
     const Plato::ScalarVector                & aState,
     const Plato::ScalarVector                & aControl,
           Plato::Scalar                        aTimeStep = 0.0) const
    {
        using ConfigScalar  = typename EvaluationType::ConfigScalarType;
        using StateScalar   = typename EvaluationType::StateScalarType;
        using ControlScalar = typename EvaluationType::ControlScalarType;
        using ResultScalar  = typename EvaluationType::ResultScalarType;

        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS);

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS);

        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS);

        auto tPairs = mSpatialModel.contactPairs();
        Plato::Contact::SurfaceDisplacementFactory<EvaluationType> tSurfaceDispFactory;
        Plato::Contact::ContactForceFactory<EvaluationType>        tContactForceFactory;

        auto tMatEntries = aInputMatrix->entries();

        for (auto tPair : tPairs)
        {
            auto computeContactForce = tContactForceFactory.create(tPair.penaltyType, tPair.penaltyValue);

            auto tSideSet = tPair.surfaceA.childSideSet();
            auto tChildCells = tPair.surfaceA.childElements();
            auto tParentCells = tPair.surfaceA.parentElements();
            auto tElementWiseChildMap = tPair.surfaceA.elementWiseChildMap();
            auto tChildFaceLocalNodes = tPair.surfaceA.childFaceLocalNodes();

            auto computeChildSurfaceDispA = tSurfaceDispFactory.createChildContribution(tPair.surfaceA);
            Plato::ScalarMultiVectorT<ResultScalar> tResultA("Results side A", mNumCells, mNumDofsPerCell);
            aFunction->evaluate_contact(mSpatialModel, tSideSet, computeChildSurfaceDispA, computeContactForce, tStateWS, tControlWS, tConfigWS, tResultA, aTimeStep);

            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, aEntryOrdinal, tResultA, tMatEntries);

            auto computeParentSurfaceDispA = tSurfaceDispFactory.createParentContribution(tPair.surfaceA, mSpatialModel.Mesh, -1.0);
            for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
            {
                computeParentSurfaceDispA->setChildNode(iChildNode);
                Plato::ScalarMultiVectorT<ResultScalar> tResultA("Results side A", mNumCells, mNumDofsPerCell);
                aFunction->evaluate_contact(mSpatialModel, tSideSet, computeParentSurfaceDispA, computeContactForce, tStateWS, tControlWS, tConfigWS, tResultA, aTimeStep);

                Plato::WorksetBase<ElementType>::assembleJacobianFad
                    (mNumDofsPerCell, tChildCells, tParentCells, tElementWiseChildMap, tChildFaceLocalNodes, iChildNode, aEntryOrdinal, tResultA, tMatEntries);
            }

            tSideSet = tPair.surfaceB.childSideSet();
            tChildCells = tPair.surfaceB.childElements();
            tParentCells = tPair.surfaceB.parentElements();
            tElementWiseChildMap = tPair.surfaceB.elementWiseChildMap();
            tChildFaceLocalNodes = tPair.surfaceB.childFaceLocalNodes();

            auto computeChildSurfaceDispB = tSurfaceDispFactory.createChildContribution(tPair.surfaceB);
            Plato::ScalarMultiVectorT<ResultScalar> tResultB("Results side B", mNumCells, mNumDofsPerCell);
            aFunction->evaluate_contact(mSpatialModel, tSideSet, computeChildSurfaceDispB, computeContactForce, tStateWS, tControlWS, tConfigWS, tResultB, aTimeStep);

            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, aEntryOrdinal, tResultB, tMatEntries);

            auto computeParentSurfaceDispB = tSurfaceDispFactory.createParentContribution(tPair.surfaceB, mSpatialModel.Mesh, -1.0);
            for (Plato::OrdinalType iChildNode = 0; iChildNode < ElementType::mNumNodesPerFace; iChildNode++)
            {
                computeParentSurfaceDispB->setChildNode(iChildNode);
                Plato::ScalarMultiVectorT<ResultScalar> tResultB("Results side B", mNumCells, mNumDofsPerCell);
                aFunction->evaluate_contact(mSpatialModel, tSideSet, computeParentSurfaceDispB, computeContactForce, tStateWS, tControlWS, tConfigWS, tResultB, aTimeStep);

                Plato::WorksetBase<ElementType>::assembleJacobianFad
                    (mNumDofsPerCell, tChildCells, tParentCells, tElementWiseChildMap, tChildFaceLocalNodes, iChildNode, aEntryOrdinal, tResultB, tMatEntries);
            }
        }
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        Plato::ScalarVector  tReturnValue("Assembled Residual",mNumDofsPerNode*mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();
            auto tInternalForceValues = this->template internalForceContribution<Residual>(mResidualFunctions.at(tName), tDomain, aState, aControl, aTimeStep);
            Plato::WorksetBase<ElementType>::assembleResidual( tInternalForceValues, tReturnValue, tDomain );
        }

        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();

        auto tExternalForceValues = this->template externalForceContribution<Residual>(mResidualFunctions.at(tFirstBlockName), aState, aControl, aTimeStep);
        Plato::WorksetBase<ElementType>::assembleResidual( tExternalForceValues, tReturnValue);

        if (mSpatialModel.hasContact())
        {
            auto tContactForceValues = this->template contactForceContribution<Residual>(mResidualFunctions.at(tFirstBlockName), aState, aControl, aTimeStep);
            Plato::WorksetBase<ElementType>::assembleResidual( tContactForceValues, tReturnValue);
        }
        
        return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        Teuchos::RCP<Plato::CrsMatrixType> tGradientXMat =
                Plato::CreateBlockMatrixTranspose<Plato::CrsMatrixType, mNumDofsPerNode, mNumSpatialDims>( mSpatialModel );
        auto tMatEntries = tGradientXMat->entries();

        Plato::BlockMatrixTransposeEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumSpatialDims>
            tGradientXMatEntryOrdinal(tGradientXMat, mSpatialModel.Mesh);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();
            auto tInternalForceValues = this->template internalForceContribution<GradientX>(mGradientXFunctions.at(tName), tDomain, aState, aControl, aTimeStep);
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumConfigDofsPerCell, tGradientXMatEntryOrdinal, tInternalForceValues, tMatEntries, tDomain);
        }

        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();

        auto tExternalForceValues = this->template externalForceContribution<GradientX>(mGradientXFunctions.at(tFirstBlockName), aState, aControl, aTimeStep);
        Plato::WorksetBase<ElementType>::assembleJacobianFad
            (mNumDofsPerCell, mNumConfigDofsPerCell, tGradientXMatEntryOrdinal, tExternalForceValues, tMatEntries);

        if (mSpatialModel.hasContact())
        {
            auto tContactForceValues = this->template contactForceContribution<GradientX>(mGradientXFunctions.at(tFirstBlockName), aState, aControl, aTimeStep);
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumConfigDofsPerCell, tGradientXMatEntryOrdinal, tContactForceValues, tMatEntries);
        }

        return tGradientXMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrixTranspose<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( mSpatialModel );
        auto tMatEntries = tJacobianMat->entries();

        Plato::BlockMatrixTransposeEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode>
            tJacobianMatEntryOrdinal( tJacobianMat, mSpatialModel.Mesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();
            auto tInternalForceValues = this->template internalForceContribution<Jacobian>(mJacobianFunctions.at(tName), tDomain, aState, aControl, aTimeStep);
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tInternalForceValues, tMatEntries, tDomain);
        }

        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();

        auto tExternalForceValues = this->template externalForceContribution<Jacobian>(mJacobianFunctions.at(tFirstBlockName), aState, aControl, aTimeStep);
        Plato::WorksetBase<ElementType>::assembleJacobianFad
            (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tExternalForceValues, tMatEntries);

        if (mSpatialModel.hasContact())
        {
            this->template contactForceNonlocalGradient <Jacobian>(mJacobianFunctions.at(tFirstBlockName), tJacobianMat, tJacobianMatEntryOrdinal, aState, aControl, aTimeStep);
        }

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( mSpatialModel );
        auto tMatEntries = tJacobianMat->entries();

        Plato::BlockMatrixEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumDofsPerNode>
            tJacobianMatEntryOrdinal( tJacobianMat, mSpatialModel.Mesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();
            auto tInternalForceValues = this->template internalForceContribution<Jacobian>(mJacobianFunctions.at(tName), tDomain, aState, aControl, aTimeStep);
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tInternalForceValues, tMatEntries, tDomain);
        }

        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();

        auto tExternalForceValues = this->template externalForceContribution<Jacobian>(mJacobianFunctions.at(tFirstBlockName), aState, aControl, aTimeStep);
        Plato::WorksetBase<ElementType>::assembleJacobianFad
            (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tExternalForceValues, tMatEntries);

        if (mSpatialModel.hasContact())
        {
            this->template contactForceNonlocalGradient <Jacobian>(mJacobianFunctions.at(tFirstBlockName), tJacobianMat, tJacobianMatEntryOrdinal, aState, aControl, aTimeStep);
        }

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        Teuchos::RCP<Plato::CrsMatrixType> tGradientZMat =
                Plato::CreateBlockMatrixTranspose<Plato::CrsMatrixType, mNumDofsPerNode, mNumControl>( mSpatialModel );
        auto tMatEntries = tGradientZMat->entries();

        Plato::BlockMatrixTransposeEntryOrdinal<mNumNodesPerCell, mNumDofsPerNode, mNumControl>
            tGradientZMatEntryOrdinal( tGradientZMat, mSpatialModel.Mesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();
            auto tInternalForceValues = this->template internalForceContribution<GradientZ>(mGradientZFunctions.at(tName), tDomain, aState, aControl, aTimeStep);
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumNodesPerCell, tGradientZMatEntryOrdinal, tInternalForceValues, tMatEntries, tDomain);
        }

        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();

        auto tExternalForceValues = this->template externalForceContribution<GradientZ>(mGradientZFunctions.at(tFirstBlockName), aState, aControl, aTimeStep);
        Plato::WorksetBase<ElementType>::assembleJacobianFad
            (mNumDofsPerCell, mNumNodesPerCell, tGradientZMatEntryOrdinal, tExternalForceValues, tMatEntries);

        if (mSpatialModel.hasContact())
        {
            auto tContactForceValues = this->template contactForceContribution<GradientZ>(mGradientZFunctions.at(tFirstBlockName), aState, aControl, aTimeStep);
            Plato::WorksetBase<ElementType>::assembleJacobianFad
                (mNumDofsPerCell, mNumNodesPerCell, tGradientZMatEntryOrdinal, tContactForceValues, tMatEntries);
        }

        return tGradientZMat;
    }
};
// class VectorFunction

} // namespace Elliptic

} // namespace Plato
