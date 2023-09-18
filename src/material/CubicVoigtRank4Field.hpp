#pragma once

#include "material/Rank4Field.hpp"
#include "material/ScalarExpression.hpp"

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

namespace Plato
{

template<typename EvaluationType>
class CubicVoigtRank4Field : public Rank4Field<EvaluationType>
{
    enum class CubicConstants { Undefined, Modulus, Lame };

protected:
    using ElementType = typename EvaluationType::ElementType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
  
public:
    CubicVoigtRank4Field(const Teuchos::ParameterList& aParams) : 
    Rank4Field<EvaluationType>(),
    mConstantsType(CubicConstants::Undefined) 
    {
        this->mStiffnessTensorProperties.clear();
        if (aParams.isSublist("Youngs Modulus")) 
        {
            this->getModulusExpressions(aParams);
            mConstantsType = CubicConstants::Modulus;
        }
        else if (aParams.isSublist("Lambda"))
        {
            this->getLameExpressions(aParams);
            mConstantsType = CubicConstants::Lame;
        }
        else
        {
            ANALYZE_THROWERR("Unrecognized representation of CubicVoigtRank4Field. Provide Modulus or Lame representations.");
        }
    }

    void
    getModulusExpressions(const Teuchos::ParameterList& aParams)
    {
        this->mStiffnessTensorProperties["Youngs Modulus"] = Plato::ScalarExpression<EvaluationType>("Youngs Modulus", aParams);
        this->mStiffnessTensorProperties["Poissons Ratio"] = Plato::ScalarExpression<EvaluationType>("Poissons Ratio", aParams);
        this->mStiffnessTensorProperties["Shear Modulus"] = Plato::ScalarExpression<EvaluationType>("Shear Modulus", aParams);
    }

    void
    getLameExpressions(const Teuchos::ParameterList& aParams)
    {
        this->mStiffnessTensorProperties["Lambda"] = Plato::ScalarExpression<EvaluationType>("Lambda", aParams);
        this->mStiffnessTensorProperties["Mu"] = Plato::ScalarExpression<EvaluationType>("Mu", aParams);
        this->mStiffnessTensorProperties["Alpha"] = Plato::ScalarExpression<EvaluationType>("Alpha", aParams);
    }

    Plato::ScalarArray4DT<KineticsScalarType>
    operator()(const Plato::ScalarMultiVectorT<ControlScalarType>& aLocalControl) override
    {
        const Plato::OrdinalType tNumCells = aLocalControl.extent(0);
        const Plato::OrdinalType tNumPoints = ElementType::getCubWeights().size();

        Plato::ScalarVectorT<ControlScalarType> tIndependentVariable("density", tNumCells*tNumPoints);
        this->calculateIndependentVariable(aLocalControl, tIndependentVariable);

        Plato::ScalarMultiVectorT<KineticsScalarType> tElementC11("C11", tNumCells, tNumPoints);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementC12("C12", tNumCells, tNumPoints);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementC44("C44", tNumCells, tNumPoints);
        if (mConstantsType == CubicConstants::Modulus)
            this->computeElementConstantsModulus(tIndependentVariable, tElementC11, tElementC12, tElementC44);
        else if (mConstantsType == CubicConstants::Lame)
            this->computeElementConstantsLame(tIndependentVariable, tElementC11, tElementC12, tElementC44);

        return this->populateStiffnessTensor(tElementC11, tElementC12, tElementC44);
    }

    void
    computeElementConstantsModulus
    (const Plato::ScalarVectorT<ControlScalarType>& aIndependentVariable,
           Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC11,
           Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC12,
           Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC44)
    {
        const Plato::OrdinalType tNumCells = aElementC11.extent(0);
        const Plato::OrdinalType tNumPoints = aElementC11.extent(1);

        Plato::ScalarMultiVectorT<KineticsScalarType> tElementYoungsModulus = this->getTensorProperty("Youngs Modulus")(aIndependentVariable);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementPoissonsRatio = this->getTensorProperty("Poissons Ratio")(aIndependentVariable);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementShearModulus = this->getTensorProperty("Shear Modulus")(aIndependentVariable);
        
        Kokkos::parallel_for("compute stiffness tensor", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),      
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            const auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;
            const auto tCurYoungsModulus = tElementYoungsModulus(tEntryOrdinal, 0);
            const auto tCurPoissonsRatio = tElementPoissonsRatio(tEntryOrdinal, 0);
            const auto tCurShearModulus = tElementShearModulus(tEntryOrdinal, 0);
            const auto tCoeff = tCurYoungsModulus / ((1.0 + tCurPoissonsRatio) * (1.0 - 2.0 * tCurPoissonsRatio));

            aElementC11(iCellOrdinal, iGpOrdinal) = tCoeff * (1.0 - tCurPoissonsRatio);
            aElementC12(iCellOrdinal, iGpOrdinal) = tCoeff * tCurPoissonsRatio;
            aElementC44(iCellOrdinal, iGpOrdinal) = tCurShearModulus;
        });
    }

    void
    computeElementConstantsLame
    (const Plato::ScalarVectorT<ControlScalarType>& aIndependentVariable,
           Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC11,
           Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC12,
           Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC44)
    {
        const Plato::OrdinalType tNumCells = aElementC11.extent(0);
        const Plato::OrdinalType tNumPoints = aElementC11.extent(1);

        Plato::ScalarMultiVectorT<KineticsScalarType> tElementLambda = this->getTensorProperty("Lambda")(aIndependentVariable);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementMu = this->getTensorProperty("Mu")(aIndependentVariable);
        Plato::ScalarMultiVectorT<KineticsScalarType> tElementAlpha = this->getTensorProperty("Alpha")(aIndependentVariable);
        
        Kokkos::parallel_for("compute stiffness tensor", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),      
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            const auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;
            const auto tCurLambda = tElementLambda(tEntryOrdinal, 0);
            const auto tCurMu = tElementMu(tEntryOrdinal, 0);
            const auto tCurAlpha = tElementAlpha(tEntryOrdinal, 0);

            aElementC11(iCellOrdinal, iGpOrdinal) = tCurLambda + 2.0 * tCurMu;
            aElementC12(iCellOrdinal, iGpOrdinal) = tCurLambda;
            aElementC44(iCellOrdinal, iGpOrdinal) = tCurAlpha;
        });
    }

    Plato::ScalarArray4DT<KineticsScalarType>
    populateStiffnessTensor
    (const Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC11,
     const Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC12,
     const Plato::ScalarMultiVectorT<KineticsScalarType>& aElementC44)
    {
        const Plato::OrdinalType tNumCells = aElementC11.extent(0);
        const Plato::OrdinalType tNumPoints = aElementC11.extent(1);

        Plato::ScalarArray4DT<KineticsScalarType> tStiffness("stiffness", tNumCells, tNumPoints, ElementType::mNumVoigtTerms, ElementType::mNumVoigtTerms);   
        
        Kokkos::parallel_for("populate stiffness tensor", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),      
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            for(int k=0; k<ElementType::mNumSpatialDims; ++k)
            {
                for(int m=0; m<ElementType::mNumSpatialDims; ++m)
                {
                    if(k==m)
                    {
                        tStiffness(iCellOrdinal, iGpOrdinal, k, m) = aElementC11(iCellOrdinal, iGpOrdinal);
                    }
                    else
                    {
                        tStiffness(iCellOrdinal, iGpOrdinal, k, m) = aElementC12(iCellOrdinal, iGpOrdinal);
                    }
                }
            }
            const int tNumShearTerms = ElementType::mNumSpatialDims*(ElementType::mNumSpatialDims-1)/2;
            for(int m=0; m<tNumShearTerms; ++m)
            {
                tStiffness(iCellOrdinal, iGpOrdinal, ElementType::mNumSpatialDims+m, ElementType::mNumSpatialDims+m) = 
                                 aElementC44(iCellOrdinal, iGpOrdinal);
            }
        });

        return tStiffness;
    }

private:
    CubicConstants mConstantsType;

};

}