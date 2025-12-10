#pragma once

#include <optional>

#include "boundary_conditions/BodyLoads.hpp"
#include "boundary_conditions/CellForcing.hpp"
#include "boundary_conditions/NaturalBCs.hpp"
#include "domain/contact/AbstractContactForce.hpp"
#include "domain/contact/AbstractSurfaceDisplacement.hpp"
#include "local_operations/optimization/ApplyWeighting.hpp"
#include "material/ElasticModelFactory.hpp"
#include "problem/elliptic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
/**
 * \brief Elastostatic vector function interface
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * \tparam IndicatorFunctionType penalty function used for density-based methods
 **********************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticResidual : public EvaluationType::ElementType,
                             public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
{
   private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumDofsPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;
    using FunctionBaseType::mSpatialDomain;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;
    Plato::CellForcing<ElementType> mCellForcing;

    std::optional<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
    std::optional<Plato::NaturalBCs<ElementType>> mBoundaryLoads;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

    std::vector<std::string> mPlotTable;

   public:
    /******************************************************************************/
    /**
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze database
     * \param [in] aProblemParams input parameters for overall problem
     * \param [in] aPenaltyParams input parameters for penalty function
     **********************************************************************************/
    ElastostaticResidual(const plato::domain::SpatialDomain& aSpatialDomain,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aProblemParams,
                         Teuchos::ParameterList& aPenaltyParams);

    /****************************************************************************/
    /**
     * \brief Pure virtual function to get output solution data
     * \param [in] state solution database
     * \return output state solution database
     ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions& aSolutions) const override;

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
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                  Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                  Plato::Scalar aTimeStep = 0.0) const override;

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
    void evaluate_boundary(const plato::domain::SpatialModel& aSpatialModel,
                           const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                           const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                           const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                           Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                           Plato::Scalar aTimeStep = 0.0) const override;

    /**********************************************************************/
    /**
     * \brief Compute Von Mises stress field and copy data into output data map
     * \param [in] aCauchyStress Cauchy stress tensor
     **************************************************************************/
    void outputVonMises(const Plato::ScalarMultiVectorT<ResultScalarType>& aCauchyStress,
                        const plato::domain::SpatialDomain& aSpatialDomain) const;
};
// class ElastostaticResidual

}  // namespace Elliptic

}  // namespace Plato
