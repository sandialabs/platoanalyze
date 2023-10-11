#pragma once

#include "AbstractMicromorphicKinetics.hpp"

#include "material/Rank4Field.hpp"
#include "material/MaterialModel.hpp"

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_RCP.hpp>

#include <memory>

namespace Plato::Hyperbolic::Micromorphic
{

template<typename EvaluationType, typename ElementType>
class ExpressionMicromorphicKinetics : public AbstractMicromorphicKinetics<EvaluationType, ElementType>
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
    using ElementType::mNumVoigtTerms;
    using ElementType::mNumSkwTerms;

public:
    ExpressionMicromorphicKinetics(const Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>> aMaterialModel) :
    AbstractMicromorphicKinetics<EvaluationType, ElementType>()
    {
        if (aMaterialModel->hasRank4Field("Ce"))
        {
            mCellMesoStressSymmetricMaterialField = aMaterialModel->template getRank4Field<EvaluationType>("Ce");
            mCellMesoStressSkewMaterialField = aMaterialModel->template getRank4Field<EvaluationType>("Cc");
            mCellMicroStressSymmetricMaterialField = aMaterialModel->template getRank4Field<EvaluationType>("Cm");
        }
        else if (aMaterialModel->hasRank4Field("Te"))
        {
            mCellMesoStressSymmetricMaterialField = aMaterialModel->template getRank4Field<EvaluationType>("Te");
            mCellMesoStressSkewMaterialField = aMaterialModel->template getRank4Field<EvaluationType>("Tc");
            mCellMicroStressSymmetricMaterialField = aMaterialModel->template getRank4Field<EvaluationType>("Jm");
            mCellMicroStressSkewMaterialField = aMaterialModel->template getRank4Field<EvaluationType>("Jc");
        }
        else
            ANALYZE_THROWERR("MaterialModel has unrecognized Rank4 Field names in ExpressionMicromorphicKinetics constructor")
    }

    void
    operator()
    (      Plato::ScalarArray3DT<KineticsScalarType>    & aSymmetricMesoStress,
           Plato::ScalarArray3DT<KineticsScalarType>    & aSkewMesoStress,
           Plato::ScalarArray3DT<KineticsScalarType>    & aSymmetricMicroStress,
     const Plato::ScalarArray3DT<KinematicsScalarType>  & aSymmetricGradientStrain,
     const Plato::ScalarArray3DT<KinematicsScalarType>  & aSkewGradientStrain,
     const Plato::ScalarArray3DT<StateScalarType>       & aSymmetricMicroStrain,
     const Plato::ScalarArray3DT<StateScalarType>       & aSkewMicroStrain,
     const Plato::ScalarMultiVectorT<ControlScalarType> & aControl) const override
    {
        const Plato::OrdinalType tNumCells = aSymmetricGradientStrain.extent(0);
        const auto tNumPoints = ElementType::getCubWeights().size();

        const auto tCellMesoStressSymmetricMaterialTensor = (*mCellMesoStressSymmetricMaterialField)(aControl);
        const auto tCellMesoStressSkewMaterialTensor = (*mCellMesoStressSkewMaterialField)(aControl);
        const auto tCellMicroStressSymmetricMaterialTensor = (*mCellMicroStressSymmetricMaterialField)(aControl);

        Kokkos::parallel_for("compute element kinematics", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCell, const Plato::OrdinalType iPoint)
        {
            for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
            {
                for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aSymmetricMesoStress(iCell,iPoint,iVoigt) += 
                        tCellMesoStressSymmetricMaterialTensor(iCell,iPoint,iVoigt,jVoigt) *
                        (aSymmetricGradientStrain(iCell,iPoint,jVoigt) - aSymmetricMicroStrain(iCell,iPoint,jVoigt));
                }
            }

            for( Plato::OrdinalType iSkew=0; iSkew<mNumSkwTerms; iSkew++)
            {
                Plato::OrdinalType StressOrdinalI = mNumSpatialDims + iSkew;
                for( Plato::OrdinalType jSkew=0; jSkew<mNumSkwTerms; jSkew++){
                    aSkewMesoStress(iCell,iPoint,StressOrdinalI) += 
                        tCellMesoStressSkewMaterialTensor(iCell,iPoint,iSkew,jSkew) *
                        (aSkewGradientStrain(iCell,iPoint,jSkew) - aSkewMicroStrain(iCell,iPoint,jSkew));
                }
            }

            for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
            {
                for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aSymmetricMicroStress(iCell,iPoint,iVoigt) += 
                        tCellMicroStressSymmetricMaterialTensor(iCell,iPoint,iVoigt,jVoigt) * 
                        aSymmetricMicroStrain(iCell,iPoint,jVoigt);
                }
            }

        });
    }

    void
    operator()
    (      Plato::ScalarArray3DT<KineticsScalarType>         & aSymmetricMesoStress,
           Plato::ScalarArray3DT<KineticsScalarType>         & aSkewMesoStress,
           Plato::ScalarArray3DT<KineticsScalarType>         & aSymmetricMicroStress,
           Plato::ScalarArray3DT<KineticsScalarType>         & aSkewMicroStress,
     const Plato::ScalarArray3DT<KinematicsDotDotScalarType> & aSymmetricGradientStrain,
     const Plato::ScalarArray3DT<KinematicsDotDotScalarType> & aSkewGradientStrain,
     const Plato::ScalarArray3DT<StateDotDotScalarType>      & aSymmetricMicroStrain,
     const Plato::ScalarArray3DT<StateDotDotScalarType>      & aSkewMicroStrain,
     const Plato::ScalarMultiVectorT<ControlScalarType>      & aControl) const override
    {
        const Plato::OrdinalType tNumCells = aSymmetricGradientStrain.extent(0);
        const auto tNumPoints = ElementType::getCubWeights().size();

        const auto tCellMesoStressSymmetricMaterialTensor = (*mCellMesoStressSymmetricMaterialField)(aControl);
        const auto tCellMesoStressSkewMaterialTensor = (*mCellMesoStressSkewMaterialField)(aControl);
        const auto tCellMicroStressSymmetricMaterialTensor = (*mCellMicroStressSymmetricMaterialField)(aControl);
        const auto tCellMicroStressSkewMaterialTensor = (*mCellMicroStressSkewMaterialField)(aControl);

        Kokkos::parallel_for("compute element kinematics", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCell, const Plato::OrdinalType iPoint)
        {
            for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
            {
                for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aSymmetricMesoStress(iCell,iPoint,iVoigt) += 
                        tCellMesoStressSymmetricMaterialTensor(iCell,iPoint,iVoigt,jVoigt) * 
                        aSymmetricGradientStrain(iCell,iPoint,jVoigt);
                }
            }

            for( Plato::OrdinalType iSkew=0; iSkew<mNumSkwTerms; iSkew++)
            {
                Plato::OrdinalType StressOrdinalI = mNumSpatialDims + iSkew;
                for( Plato::OrdinalType jSkew=0; jSkew<mNumSkwTerms; jSkew++){
                    aSkewMesoStress(iCell,iPoint,StressOrdinalI) += 
                        tCellMesoStressSkewMaterialTensor(iCell,iPoint,iSkew,jSkew) *
                        aSkewGradientStrain(iCell,iPoint,jSkew);
                }
            }

            for( Plato::OrdinalType iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++)
            {
                for( Plato::OrdinalType jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aSymmetricMicroStress(iCell,iPoint,iVoigt) += 
                        tCellMicroStressSymmetricMaterialTensor(iCell,iPoint,iVoigt,jVoigt) *
                        aSymmetricMicroStrain(iCell,iPoint,jVoigt);
                }
            }

            for( Plato::OrdinalType iSkew=0; iSkew<mNumSkwTerms; iSkew++)
            {
                Plato::OrdinalType StressOrdinalI = mNumSpatialDims + iSkew;
                for( Plato::OrdinalType jSkew=0; jSkew<mNumSkwTerms; jSkew++){
                    aSkewMicroStress(iCell,iPoint,StressOrdinalI) += 
                        tCellMicroStressSkewMaterialTensor(iCell,iPoint,iSkew,jSkew) *
                        aSkewMicroStrain(iCell,iPoint,jSkew);
                }
            }

        });
    }

private:
    std::shared_ptr<Plato::Rank4Field<EvaluationType>> mCellMesoStressSymmetricMaterialField;
    std::shared_ptr<Plato::Rank4Field<EvaluationType>> mCellMesoStressSkewMaterialField;
    std::shared_ptr<Plato::Rank4Field<EvaluationType>> mCellMicroStressSymmetricMaterialField;
    std::shared_ptr<Plato::Rank4Field<EvaluationType>> mCellMicroStressSkewMaterialField;
};

}