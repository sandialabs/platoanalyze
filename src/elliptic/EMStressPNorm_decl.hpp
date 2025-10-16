#pragma once

#include "ApplyWeighting.hpp"
#include "LinearElectroelasticMaterial.hpp"
#include "TensorPNorm.hpp"
#include "elliptic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template <typename EvaluationType, typename IndicatorFunctionType>
class EMStressPNorm : public EvaluationType::ElementType, public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
   private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumDofsPerCell;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    using FunctionBaseType = typename Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mDataMap;
    using FunctionBaseType::mSpatialDomain;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<mNumSpatialDims>> mMaterialModel;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mNumNodesPerCell, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms, EvaluationType>> mNorm;

    std::string mFuncString = "1.0";

   public:
    /**************************************************************************/
    EMStressPNorm(const Plato::SpatialDomain& aSpatialDomain,
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
// class EMStressPNorm

}  // namespace Elliptic

}  // namespace Plato
