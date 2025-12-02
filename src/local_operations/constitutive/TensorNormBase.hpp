#ifndef PLATO_LOCAL_OPERATIONS_CONSTITUTIVE_TENSORNORMBASE
#define PLATO_LOCAL_OPERATIONS_CONSTITUTIVE_TENSORNORMBASE

#include <variant>

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "local_operations/constitutive/VoigtTensorL2Norm.hpp"
#include "local_operations/constitutive/VonMisesNorm.hpp"

namespace Plato
{
/// @brief abstract class for computing norms of a tensor in Voigt notation
/// @tparam VoigtLength size of tensor
/// @tparam EvaluationType struct containing the FAD types defining which derivatives to take with AD
template <Plato::OrdinalType VoigtLength, typename EvaluationType>
class TensorNormBase
{
   private:
    using ResultScalarType = typename EvaluationType::ResultScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using NormStrategyVariant = std::variant<VoigtTensorL2Norm, VonMisesNorm>;

   public:
    template <typename NormStrategy>
    TensorNormBase(const Plato::Scalar aExponent, const bool aScaleByVolume, NormStrategy aNormStrategy);

    /// @brief compute the tensor norm of @a aElementAveragedVoigtStresses raised to the exponent for all elements in @a
    /// aReturnValues
    void evaluate(Plato::ScalarVectorT<ResultScalarType> aReturnValues,
                  const Plato::ScalarMultiVectorT<ResultScalarType> aElementAveragedVoigtStresses,
                  const Plato::ScalarMultiVectorT<ControlScalarType> aControl,
                  const Plato::ScalarVectorT<ConfigScalarType> aCellVolumes) const;

    /// @brief scale all entries in @a aResultSensitivities by the derivative of the outer p norm (i.e. scale factor is
    /// d/df[(f)^(1/p)] = 1/p * f^((1-p)/p))
    void postEvaluate(Plato::ScalarVector aResultSensitivities, const Plato::Scalar aResultValue) const;

    /// @brief evaluate f^(1/p) with f as @a aResultValue
    void postEvaluate(Plato::Scalar& aResultValue) const;

   private:
    /// @tparam NormStrategy struct containing a kokkos inlined operator() that evaluates the specific norm
    template <typename NormStrategy>
    void evaluateImpl(Plato::ScalarVectorT<ResultScalarType> aReturnValues,
                      const Plato::ScalarMultiVectorT<ResultScalarType> aElementAveragedVoigtStresses,
                      const Plato::ScalarMultiVectorT<ControlScalarType> aControl,
                      const Plato::ScalarVectorT<ConfigScalarType> aCellVolumes,
                      const NormStrategy& aNormStrategy) const;

   private:
    Plato::Scalar mExponent;
    bool mScaleByVolume;
    NormStrategyVariant mNormStrategyVariant;
};

template <Plato::OrdinalType VoigtLength, typename EvaluationType>
template <typename NormStrategy>
TensorNormBase<VoigtLength, EvaluationType>::TensorNormBase(const Plato::Scalar aExponent,
                                                            const bool aScaleByVolume,
                                                            NormStrategy aNormStrategy)
    : mExponent{aExponent}, mScaleByVolume{aScaleByVolume}, mNormStrategyVariant{std::move(aNormStrategy)}
{
}

template <Plato::OrdinalType VoigtLength, typename EvaluationType>
void TensorNormBase<VoigtLength, EvaluationType>::evaluate(
    Plato::ScalarVectorT<ResultScalarType> aReturnValues,
    const Plato::ScalarMultiVectorT<ResultScalarType> aElementAveragedVoigtStresses,
    const Plato::ScalarMultiVectorT<ControlScalarType> aControl,
    const Plato::ScalarVectorT<ConfigScalarType> aCellVolumes) const
{
    std::visit([&](const auto& aNormStrategy)
               { evaluateImpl(aReturnValues, aElementAveragedVoigtStresses, aControl, aCellVolumes, aNormStrategy); },
               mNormStrategyVariant);
}

template <Plato::OrdinalType VoigtLength, typename EvaluationType>
void TensorNormBase<VoigtLength, EvaluationType>::postEvaluate(Plato::ScalarVector aResultSensitivities,
                                                               const Plato::Scalar aResultValue) const
{
    const auto tScaleFactor = pow(aResultValue, (1.0 - mExponent) / mExponent) / mExponent;
    const auto tNumEntries = aResultSensitivities.size();
    Kokkos::parallel_for(
        "scale vector", Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumEntries),
        KOKKOS_LAMBDA(Plato::OrdinalType aEntryOrdinal) { aResultSensitivities(aEntryOrdinal) *= tScaleFactor; });
}

template <Plato::OrdinalType VoigtLength, typename EvaluationType>
void TensorNormBase<VoigtLength, EvaluationType>::postEvaluate(Plato::Scalar& aResultValue) const
{
    aResultValue = pow(aResultValue, 1.0 / mExponent);
}

template <Plato::OrdinalType VoigtLength, typename EvaluationType>
template <typename NormStrategy>
void TensorNormBase<VoigtLength, EvaluationType>::evaluateImpl(
    Plato::ScalarVectorT<ResultScalarType> aReturnValues,
    const Plato::ScalarMultiVectorT<ResultScalarType> aElementAveragedVoigtStresses,
    const Plato::ScalarMultiVectorT<ControlScalarType> aControl,
    const Plato::ScalarVectorT<ConfigScalarType> aCellVolumes,
    const NormStrategy& aNormStrategy) const
{
    const Plato::OrdinalType tNumCells = aReturnValues.extent(0);
    const auto tExponent = mExponent;
    const auto tScaleByVolume = mScaleByVolume;
    Kokkos::parallel_for(
        "Compute tensor norm", Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumCells),
        KOKKOS_LAMBDA(Plato::OrdinalType tCellOrdinal) {
            Plato::Array<VoigtLength, ResultScalarType> tStressValues{0.0};
            for (Plato::OrdinalType iVoigt = 0; iVoigt < VoigtLength; iVoigt++)
            {
                tStressValues(iVoigt) = aElementAveragedVoigtStresses(tCellOrdinal, iVoigt);
            }
            ResultScalarType tStressNorm = aNormStrategy(tStressValues);

            aReturnValues(tCellOrdinal) = pow(tStressNorm, tExponent);
            if (tScaleByVolume)
            {
                aReturnValues(tCellOrdinal) *= aCellVolumes(tCellOrdinal);
            }
        });
}

}  // namespace Plato

#endif
