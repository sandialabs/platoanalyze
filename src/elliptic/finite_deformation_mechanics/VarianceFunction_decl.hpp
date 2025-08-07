#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_VARIANCEFUNCTION_DECL_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_VARIANCEFUNCTION_DECL_H

#include <Teuchos_ParameterList.hpp>
#include <map>
#include <memory>
#include <string>

#include "PlatoStaticsTypes.hpp"
#include "PlatoTypes.hpp"
#include "Solutions.hpp"
#include "SpatialModel.hpp"
#include "WorksetBase.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/EvaluationTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
/// @brief class that computes the variance of quantities defined at the element level.
/// This class has the responsibility of constructing and calling AbstractScalarFunction implementations to compute
/// local (element-level) quantities with the correct Sacado FAD types and using them to compute the overall gradients.
/// @tparam PhysicsType struct specifying element and function factory types for the specific class of physics.
template <typename PhysicsType>
class VarianceFunction : public Plato::Elliptic::ScalarFunctionBase,
                         public Plato::WorksetBase<typename PhysicsType::ElementType>
{
   private:
    using ElementType = typename PhysicsType::ElementType;
    using Plato::WorksetBase<ElementType>::mNumDofsPerCell;
    using Plato::WorksetBase<ElementType>::mNumNodesPerCell;
    using Plato::WorksetBase<ElementType>::mNumSpatialDims;
    using Plato::WorksetBase<ElementType>::mNumDofsPerNode;
    using Plato::WorksetBase<ElementType>::mNumNodes;

    using Plato::WorksetBase<ElementType>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<ElementType>::mControlEntryOrdinal;
    using Plato::WorksetBase<ElementType>::mConfigEntryOrdinal;

    template <typename EvaluationType>
    using EvaluationFunctionMap =
        std::map<std::string, std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>>;

    using Residual = typename Plato::Elliptic::Evaluation<ElementType>::Residual;
    using Jacobian = typename Plato::Elliptic::Evaluation<ElementType>::Jacobian;
    using GradientX = typename Plato::Elliptic::Evaluation<ElementType>::GradientX;
    using GradientZ = typename Plato::Elliptic::Evaluation<ElementType>::GradientZ;

   public:
    VarianceFunction(const Plato::SpatialModel& aSpatialModel,
                     Plato::DataMap& aDataMap,
                     Teuchos::ParameterList& aProblemParams,
                     const std::string& aName);

    /// @brief Return user defined function name
    std::string name() const override final;

    /// @brief Compute the variance of local quantities
    Plato::Scalar value(const Plato::Solutions& aSolution,
                        const Plato::ScalarVector& aControl,
                        Plato::Scalar aTimeStep = 0.0) const override final;

    /// @brief Compute the gradient of the variance of local quantities w.r.t. control
    Plato::ScalarVector gradient_z(const Plato::Solutions& aSolution,
                                   const Plato::ScalarVector& aControl,
                                   Plato::Scalar aTimeStep = 0.0) const override final;

    /// @brief Compute the gradient of the variance of local quantities w.r.t. state
    Plato::ScalarVector gradient_u(const Plato::Solutions& aSolution,
                                   const Plato::ScalarVector& aControl,
                                   Plato::OrdinalType aStepIndex,
                                   Plato::Scalar aTimeStep = 0.0) const override final;

    /// @brief Compute the gradient of the variance of local quantities w.r.t. config (nodal coordinates)
    Plato::ScalarVector gradient_x(const Plato::Solutions& aSolution,
                                   const Plato::ScalarVector& aControl,
                                   Plato::Scalar aTimeStep = 0.0) const override final;

    /// @brief Update parameters such as penalty values during optimization iterations. This function appears to be
    /// depreciated
    void updateProblem(const Plato::ScalarVector& aState, const Plato::ScalarVector& aControl) const override final;

   private:
    template <typename EvaluationType>
    auto computeElementWiseField(const Plato::ScalarVector& aState,
                                 const Plato::ScalarVector& aControl,
                                 const Plato::Scalar aTimeStep,
                                 const EvaluationFunctionMap<EvaluationType>& aFunctionMap) const
        -> std::map<std::string, Plato::ScalarVectorT<typename EvaluationType::ResultScalarType>>;

   private:
    Plato::SpatialModel mSpatialModel;
    Plato::DataMap mDataMap;
    std::string mFunctionName;
    Plato::OrdinalType mNumTotalCells;
    EvaluationFunctionMap<Residual> mValueFunctions;
    EvaluationFunctionMap<Jacobian> mGradientUFunctions;
    EvaluationFunctionMap<GradientX> mGradientXFunctions;
    EvaluationFunctionMap<GradientZ> mGradientZFunctions;
};

namespace detail
{
Plato::ScalarVector get_last_time_step_state(const Plato::Solutions& aSolution);

template <typename ResultScalarType>
Plato::Scalar compute_field_mean(const Plato::SpatialModel& aSpatialModel,
                                 const std::map<std::string, Plato::ScalarVectorT<ResultScalarType>>& aDomainResults,
                                 const Plato::OrdinalType aNumTotalCells);

template <>
Plato::Scalar compute_field_mean(const Plato::SpatialModel& aSpatialModel,
                                 const std::map<std::string, Plato::ScalarVectorT<Plato::Scalar>>& aDomainResults,
                                 const Plato::OrdinalType aNumTotalCells);

Plato::Scalar compute_field_variance(const Plato::SpatialModel& aSpatialModel,
                                     const std::map<std::string, Plato::ScalarVectorT<Plato::Scalar>>& aDomainResults,
                                     const Plato::Scalar aMean,
                                     const Plato::OrdinalType aNumTotalCells);

template <typename ResultScalarType>
void scale_result_by_variance_derivative(
    const Plato::SpatialModel& aSpatialModel,
    const std::map<std::string, Plato::ScalarVectorT<ResultScalarType>>& aDomainResults,
    const Plato::Scalar aMean,
    const Plato::OrdinalType aNumTotalCells);
}  // namespace detail
}  // namespace plato::elliptic::finite_deformation_mechanics
#endif
