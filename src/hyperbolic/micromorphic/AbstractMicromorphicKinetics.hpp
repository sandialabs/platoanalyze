#pragma once

#include "FadTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "material/MaterialModel.hpp"

#include <Teuchos_RCP.hpp>

namespace Plato::Hyperbolic::Micromorphic
{

template<typename EvaluationType, typename ElementType>
class AbstractMicromorphicKinetics : public ElementType
{
protected:
    using StateScalarType  = typename EvaluationType::StateScalarType;
    using StateDotDotScalarType  = typename EvaluationType::StateDotDotScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using KinematicsScalarType = typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;
    using KinematicsDotDotScalarType = typename Plato::fad_type_t<ElementType, StateDotDotScalarType, ConfigScalarType>;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    using ElementType::mNumSpatialDims;

public:
    AbstractMicromorphicKinetics() = default;

    virtual ~AbstractMicromorphicKinetics() = default;

    AbstractMicromorphicKinetics(const AbstractMicromorphicKinetics& aKinetics) = delete;

    AbstractMicromorphicKinetics(AbstractMicromorphicKinetics&& aKinetics) = delete;

    AbstractMicromorphicKinetics&
    operator=(const AbstractMicromorphicKinetics& aKinetics) = delete;

    AbstractMicromorphicKinetics&
    operator=(AbstractMicromorphicKinetics&& aKinetics) = delete;

    virtual void
    operator()
    (      Plato::ScalarArray3DT<KineticsScalarType>    & aSymmetricMesoStress,
           Plato::ScalarArray3DT<KineticsScalarType>    & aSkewMesoStress,
           Plato::ScalarArray3DT<KineticsScalarType>    & aSymmetricMicroStress,
     const Plato::ScalarArray3DT<KinematicsScalarType>  & aSymmetricGradientStrain,
     const Plato::ScalarArray3DT<KinematicsScalarType>  & aSkewGradientStrain,
     const Plato::ScalarArray3DT<StateScalarType>       & aSymmetricMicroStrain,
     const Plato::ScalarArray3DT<StateScalarType>       & aSkewMicroStrain,
     const Plato::ScalarMultiVectorT<ControlScalarType> & aControl) const = 0;

    virtual void
    operator()
    (      Plato::ScalarArray3DT<KineticsScalarType>         & aSymmetricMesoStress,
           Plato::ScalarArray3DT<KineticsScalarType>         & aSkewMesoStress,
           Plato::ScalarArray3DT<KineticsScalarType>         & aSymmetricMicroStress,
           Plato::ScalarArray3DT<KineticsScalarType>         & aSkewMicroStress,
     const Plato::ScalarArray3DT<KinematicsDotDotScalarType> & aSymmetricGradientStrain,
     const Plato::ScalarArray3DT<KinematicsDotDotScalarType> & aSkewGradientStrain,
     const Plato::ScalarArray3DT<StateDotDotScalarType>      & aSymmetricMicroStrain,
     const Plato::ScalarArray3DT<StateDotDotScalarType>      & aSkewMicroStrain,
     const Plato::ScalarMultiVectorT<ControlScalarType>      & aControl) const = 0;
};

}