#pragma once

#include "linear_algebra/TensorPNorm.hpp"
#include "local_operations/optimization/ApplyWeighting.hpp"
#include "material/ElasticModelFactory.hpp"
#include "problem/elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
class StressPNorm : public EvaluationType::ElementType, public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
   private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumDofsPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms, EvaluationType>> mNorm;

    std::string mFuncString = "1.0";

    Plato::ScalarMultiVector mFxnValues;

    Teuchos::RCP<Plato::LinearElasticMaterial<mNumSpatialDims>> mMaterialModel;

   public:
    /**************************************************************************/
    StressPNorm(const Plato::SpatialDomain& aSpatialDomain,
                Plato::DataMap& aDataMap,
                Teuchos::ParameterList& aProblemParams,
                Teuchos::ParameterList& aPenaltyParams,
                const std::string& aFunctionName);

    /**************************************************************************/
    void evaluate_conditional(const Plato::ScalarMultiVectorT<StateScalarType>& aState,
                              const Plato::ScalarMultiVectorT<ControlScalarType>& aControl,
                              const Plato::ScalarArray3DT<ConfigScalarType>& aConfig,
                              Plato::ScalarVectorT<ResultScalarType>& aResult,
                              Plato::Scalar aTimeStep = 0.0) const override;

    /**************************************************************************/
    void postEvaluate(Plato::ScalarVector resultVector, Plato::Scalar resultScalar) override;
    /**************************************************************************/

    /**************************************************************************/
    void postEvaluate(Plato::Scalar& resultValue) override;
    /**************************************************************************/
};
// class StressPNorm

}  // namespace Elliptic

}  // namespace Plato
