#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_VARIANCEFUNCTION_DEF_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_VARIANCEFUNCTION_DEF_H

#include <Teuchos_ParameterList.hpp>
#include <map>
#include <string>

#include "AnalyzeMacros.hpp"
#include "BLAS1.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTypes.hpp"
#include "SpatialModel.hpp"
#include "elliptic/finite_deformation_mechanics/VarianceFunction_decl.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
template <typename PhysicsType>
VarianceFunction<PhysicsType>::VarianceFunction(const Plato::SpatialModel& aSpatialModel,
                                                Plato::DataMap& aDataMap,
                                                Teuchos::ParameterList& aProblemParams,
                                                const std::string& aName)
    : Plato::WorksetBase<ElementType>(aSpatialModel.Mesh),
      mSpatialModel(aSpatialModel),
      mDataMap(aDataMap),
      mFunctionName(aName),
      mNumTotalCells{0}
{
    typename PhysicsType::FunctionFactory tFactory;

    auto tProblemDefault = aProblemParams.sublist("Criteria").sublist(mFunctionName);
    auto tFieldVariable = tProblemDefault.get<std::string>("Field Variable", "");

    for (const auto& tDomain : mSpatialModel.Domains)
    {
        auto tName = tDomain.getDomainName();
        mNumTotalCells += tDomain.numCells();

        mValueFunctions[tName] = tFactory.template createScalarFunction<Residual>(tDomain, mDataMap, aProblemParams,
                                                                                  tFieldVariable, mFunctionName);
        mGradientUFunctions[tName] = tFactory.template createScalarFunction<Jacobian>(tDomain, mDataMap, aProblemParams,
                                                                                      tFieldVariable, mFunctionName);
        mGradientXFunctions[tName] = tFactory.template createScalarFunction<GradientX>(
            tDomain, mDataMap, aProblemParams, tFieldVariable, mFunctionName);
        mGradientZFunctions[tName] = tFactory.template createScalarFunction<GradientZ>(
            tDomain, mDataMap, aProblemParams, tFieldVariable, mFunctionName);
    }
}

template <typename PhysicsType>
std::string VarianceFunction<PhysicsType>::name() const
{
    return mFunctionName;
}

template <typename PhysicsType>
Plato::Scalar VarianceFunction<PhysicsType>::value(const Plato::Solutions& aSolution,
                                                   const Plato::ScalarVector& aControl,
                                                   Plato::Scalar aTimeStep) const
{
    const auto tFinalState = detail::get_last_time_step_state(aSolution);

    const auto tDomainResults = computeElementWiseField<Residual>(tFinalState, aControl, aTimeStep, mValueFunctions);

    const auto tMean = detail::compute_field_mean<Plato::Scalar>(mSpatialModel, tDomainResults, mNumTotalCells);
    return detail::compute_field_variance(mSpatialModel, tDomainResults, tMean, mNumTotalCells);
}

template <typename PhysicsType>
Plato::ScalarVector VarianceFunction<PhysicsType>::gradient_z(const Plato::Solutions& aSolution,
                                                              const Plato::ScalarVector& aControl,
                                                              Plato::Scalar aTimeStep) const
{
    const auto tFinalState = detail::get_last_time_step_state(aSolution);

    const auto tDomainResults =
        computeElementWiseField<GradientZ>(tFinalState, aControl, aTimeStep, mGradientZFunctions);

    const auto tMean =
        detail::compute_field_mean<typename GradientZ::ResultScalarType>(mSpatialModel, tDomainResults, mNumTotalCells);

    detail::scale_result_by_variance_derivative<typename GradientZ::ResultScalarType>(mSpatialModel, tDomainResults,
                                                                                      tMean, mNumTotalCells);

    // sum gradient entries w.r.t. the same DOFs
    Plato::ScalarVector tGradient("gradient vector", mNumNodes);
    for (const auto& tDomain : mSpatialModel.Domains)
    {
        const auto tResult = tDomainResults.at(tDomain.getDomainName());
        Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>(tDomain, mControlEntryOrdinal, tResult, tGradient);
    }
    return tGradient;
}

template <typename PhysicsType>
Plato::ScalarVector VarianceFunction<PhysicsType>::gradient_u(const Plato::Solutions& aSolution,
                                                              const Plato::ScalarVector& aControl,
                                                              Plato::OrdinalType aStepIndex,
                                                              Plato::Scalar aTimeStep) const
{
    const auto tFinalState = detail::get_last_time_step_state(aSolution);

    const auto tDomainResults =
        computeElementWiseField<Jacobian>(tFinalState, aControl, aTimeStep, mGradientUFunctions);

    const auto tMean =
        detail::compute_field_mean<typename Jacobian::ResultScalarType>(mSpatialModel, tDomainResults, mNumTotalCells);

    detail::scale_result_by_variance_derivative<typename Jacobian::ResultScalarType>(mSpatialModel, tDomainResults,
                                                                                     tMean, mNumTotalCells);

    // sum gradient entries w.r.t. the same DOFs
    Plato::ScalarVector tGradient("gradient vector", mNumDofsPerNode * mNumNodes);
    for (const auto& tDomain : mSpatialModel.Domains)
    {
        const auto tResult = tDomainResults.at(tDomain.getDomainName());
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(tDomain, mGlobalStateEntryOrdinal,
                                                                               tResult, tGradient);
    }
    return tGradient;
}

template <typename PhysicsType>
Plato::ScalarVector VarianceFunction<PhysicsType>::gradient_x(const Plato::Solutions& aSolution,
                                                              const Plato::ScalarVector& aControl,
                                                              Plato::Scalar aTimeStep) const
{
    const auto tFinalState = detail::get_last_time_step_state(aSolution);

    const auto tDomainResults =
        computeElementWiseField<GradientX>(tFinalState, aControl, aTimeStep, mGradientXFunctions);

    const auto tMean =
        detail::compute_field_mean<typename GradientX::ResultScalarType>(mSpatialModel, tDomainResults, mNumTotalCells);

    detail::scale_result_by_variance_derivative<typename GradientX::ResultScalarType>(mSpatialModel, tDomainResults,
                                                                                      tMean, mNumTotalCells);

    // sum gradient entries w.r.t. the same DOFs
    Plato::ScalarVector tGradient("gradient vector", mNumSpatialDims * mNumNodes);
    for (const auto& tDomain : mSpatialModel.Domains)
    {
        const auto tResult = tDomainResults.at(tDomain.getDomainName());
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>(tDomain, mConfigEntryOrdinal, tResult,
                                                                               tGradient);
    }
    return tGradient;
}

template <typename PhysicsType>
void VarianceFunction<PhysicsType>::updateProblem(const Plato::ScalarVector& aState,
                                                  const Plato::ScalarVector& aControl) const
{
    ANALYZE_THROWERR("updateProblem has not been implemented in VarianceFunction class.")
}

template <typename PhysicsType>
template <typename EvaluationType>
auto VarianceFunction<PhysicsType>::computeElementWiseField(const Plato::ScalarVector& aState,
                                                            const Plato::ScalarVector& aControl,
                                                            const Plato::Scalar aTimeStep,
                                                            const EvaluationFunctionMap<EvaluationType>& aFunctionMap)
    const -> std::map<std::string, Plato::ScalarVectorT<typename EvaluationType::ResultScalarType>>
{
    using ConfigScalar = typename EvaluationType::ConfigScalarType;
    using StateScalar = typename EvaluationType::StateScalarType;
    using ControlScalar = typename EvaluationType::ControlScalarType;
    using ResultScalar = typename EvaluationType::ResultScalarType;

    std::map<std::string, Plato::ScalarVectorT<ResultScalar>> tDomainResults;
    for (const auto& tDomain : mSpatialModel.Domains)
    {
        const auto tNumCells = tDomain.numCells();
        const auto tName = tDomain.getDomainName();

        // workset state
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", tNumCells, mNumDofsPerCell);
        Plato::WorksetBase<ElementType>::worksetState(aState, tStateWS, tDomain);

        // workset control
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
        Plato::WorksetBase<ElementType>::worksetControl(aControl, tControlWS, tDomain);

        // workset config
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<ElementType>::worksetConfig(tConfigWS, tDomain);

        // create result view
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResult);

        // evaluate function
        aFunctionMap.at(tName)->evaluate(tStateWS, tControlWS, tConfigWS, tResult, aTimeStep);
        tDomainResults[tName] = tResult;
    }
    return tDomainResults;
}

namespace detail
{
template <typename ResultScalarType>
Plato::Scalar compute_field_mean(const Plato::SpatialModel& aSpatialModel,
                                 const std::map<std::string, Plato::ScalarVectorT<ResultScalarType>>& aDomainResults,
                                 const Plato::OrdinalType aNumTotalCells)
{
    Plato::Scalar tMean{0.0};
    for (const auto& tDomain : aSpatialModel.Domains)
    {
        const auto tNumCells = tDomain.numCells();
        const auto tResult = aDomainResults.at(tDomain.getDomainName());
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<>(0, tNumCells),
            KOKKOS_LAMBDA(const Plato::OrdinalType tCellOrdinal, Plato::Scalar& aUpdate) {
                aUpdate += tResult(tCellOrdinal).val();
            },
            tMean);
    }
    return tMean / aNumTotalCells;
}

template <>
Plato::Scalar compute_field_mean(const Plato::SpatialModel& aSpatialModel,
                                 const std::map<std::string, Plato::ScalarVectorT<Plato::Scalar>>& aDomainResults,
                                 const Plato::OrdinalType aNumTotalCells)
{
    Plato::Scalar tMean{0.0};
    for (const auto& tDomain : aSpatialModel.Domains)
    {
        const auto tNumCells = tDomain.numCells();
        const auto tResult = aDomainResults.at(tDomain.getDomainName());
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<>(0, tNumCells),
            KOKKOS_LAMBDA(const Plato::OrdinalType tCellOrdinal, Plato::Scalar& aUpdate) {
                aUpdate += tResult(tCellOrdinal);
            },
            tMean);
    }
    return tMean / aNumTotalCells;
}

template <typename ResultScalarType>
void scale_result_by_variance_derivative(
    const Plato::SpatialModel& aSpatialModel,
    const std::map<std::string, Plato::ScalarVectorT<ResultScalarType>>& aDomainResults,
    const Plato::Scalar aMean,
    const Plato::OrdinalType aNumTotalCells)
{
    for (const auto& tDomain : aSpatialModel.Domains)
    {
        const auto tNumCells = tDomain.numCells();
        const auto tResult = aDomainResults.at(tDomain.getDomainName());
        Kokkos::parallel_for(
            "scale by variance derivative", Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumCells),
            KOKKOS_LAMBDA(const Plato::OrdinalType tCellOrdinal) {
                tResult(tCellOrdinal) *= 2 * (tResult(tCellOrdinal).val() - aMean) / aNumTotalCells;
            });
    }
}
}  // namespace detail
}  // namespace plato::elliptic::finite_deformation_mechanics
#endif
