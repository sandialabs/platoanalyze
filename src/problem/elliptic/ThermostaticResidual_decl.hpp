#pragma once

#include <optional>

#include "boundary_conditions/BodyLoads.hpp"
#include "boundary_conditions/NaturalBCs.hpp"
#include "domain/contact/AbstractContactForce.hpp"
#include "domain/contact/AbstractSurfaceDisplacement.hpp"
#include "local_operations/optimization/ApplyWeighting.hpp"
#include "material/ThermalConductivityMaterial.hpp"
#include "problem/elliptic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
class ThermostaticResidual : public EvaluationType::ElementType,
                             public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
   private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumDofsPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = Plato::Elliptic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mDofNames;
    using FunctionBaseType::mSpatialDomain;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyWeighting;

    std::optional<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
    std::optional<Plato::NaturalBCs<ElementType, mNumDofsPerNode>> mBoundaryLoads;

    Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> mMaterialModel;

    std::vector<std::string> mPlottable;

   public:
    /**************************************************************************/
    ThermostaticResidual(const Plato::SpatialDomain& aSpatialDomain,
                         Plato::DataMap& aDataMap,
                         Teuchos::ParameterList& aProblemParams,
                         Teuchos::ParameterList& penaltyParams);

    /****************************************************************************/
    /**
     * \brief Pure virtual function to get output solution data
     * \param [in] state solution database
     * \return output state solution database
     ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions& aSolutions) const override;

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                  Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                  Plato::Scalar aTimeStep = 0.0) const override;

    /**************************************************************************/
    void evaluate_boundary(const Plato::SpatialModel& aSpatialModel,
                           const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                           const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                           const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                           Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                           Plato::Scalar aTimeStep = 0.0) const override;
};
// class ThermostaticResidual

}  // namespace Elliptic

}  // namespace Plato
