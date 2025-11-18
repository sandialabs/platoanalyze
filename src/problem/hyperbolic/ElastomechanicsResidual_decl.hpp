#pragma once

#include <optional>

#include "boundary_conditions/BodyLoads.hpp"
#include "boundary_conditions/NaturalBCs.hpp"
#include "local_operations/optimization/ApplyWeighting.hpp"
#include "material/ElasticModelFactory.hpp"
#include "problem/hyperbolic/EvaluationTypes.hpp"
#include "problem/hyperbolic/VectorFunction.hpp"

namespace Plato
{

namespace Hyperbolic
{

template <typename EvaluationType, typename IndicatorFunctionType>
class TransientMechanicsResidual : public EvaluationType::ElementType,
                                   public Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>
{
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumDofsPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    using FunctionBaseType = Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>;

    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mSpatialDomain;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumSpatialDims, IndicatorFunctionType> mApplyMassWeighting;

    std::optional<Plato::BodyLoads<EvaluationType, ElementType>> mBodyLoads;
    std::optional<Plato::NaturalBCs<ElementType>> mBoundaryLoads;

    bool mRayleighDamping;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

    std::vector<std::string> mPlotTable;

   public:
    TransientMechanicsResidual(const Plato::SpatialDomain& aSpatialDomain,
                               Plato::DataMap& aDataMap,
                               Teuchos::ParameterList& aProblemParams,
                               Teuchos::ParameterList& aPenaltyParams);

    Plato::Scalar getMaxEigenvalue(const Plato::ScalarArray3D& aConfig) const override;

    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions& aSolutions) const override;

    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                  const Plato::ScalarMultiVectorT<StateDotScalarType>& aStateDot,
                  const Plato::ScalarMultiVectorT<StateDotDotScalarType>& aStateDotDot,
                  const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                  Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                  Plato::Scalar aTimeStep = 0.0,
                  Plato::Scalar aCurrentTime = 0.0) const override;

    void evaluateWithoutDamping(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                                const Plato::ScalarMultiVectorT<StateDotScalarType>& aStateDot,
                                const Plato::ScalarMultiVectorT<StateDotDotScalarType>& aStateDotDot,
                                const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                                const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                                Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                                Plato::Scalar aTimeStep = 0.0,
                                Plato::Scalar aCurrentTime = 0.0) const;

    void evaluateWithDamping(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                             const Plato::ScalarMultiVectorT<StateDotScalarType>& aStateDot,
                             const Plato::ScalarMultiVectorT<StateDotDotScalarType>& aStateDotDot,
                             const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                             const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                             Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                             Plato::Scalar aTimeStep = 0.0,
                             Plato::Scalar aCurrentTime = 0.0) const;

    void evaluate_boundary(const Plato::SpatialModel& aSpatialModel,
                           const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                           const Plato::ScalarMultiVectorT<StateDotScalarType>& aStateDot,
                           const Plato::ScalarMultiVectorT<StateDotDotScalarType>& aStateDotDot,
                           const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                           const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                           Plato::ScalarMultiVectorT<ResultScalarType>& aResult,
                           Plato::Scalar aTimeStep = 0.0,
                           Plato::Scalar aCurrentTime = 0.0) const override;
};

}  // namespace Hyperbolic

}  // namespace Plato
