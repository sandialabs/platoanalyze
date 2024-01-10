/*
 * DynamicCompliance.hpp
 *
 *  Created on: Apr 25, 2018
 */

#ifndef DYNAMICCOMPLIANCE_HPP_
#define DYNAMICCOMPLIANCE_HPP_

#include <memory>

#include "PlatoMathTypes.hpp"

#include <Teuchos_ParameterList.hpp>


#include "StateValues.hpp"
#include "ApplyPenalty.hpp"
#include "ComplexStrain.hpp"
#include "SimplexFadTypes.hpp"
#include "ApplyProjection.hpp"
#include "ImplicitFunctors.hpp"
#include "PlatoStaticsTypes.hpp"
#include "ComplexLinearStress.hpp"
#include "ElasticModelFactory.hpp"
#include "ComplexElasticEnergy.hpp"
#include "ComplexInertialEnergy.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "SimplexStructuralDynamics.hpp"
#include "IsotropicLinearElasticMaterial.hpp"

namespace Plato
{

template<typename EvaluationType, class PenaltyFuncType, class ProjectionFuncType>
class DynamicCompliance:
        public Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim, EvaluationType::NumControls>,
        public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::mNumVoigtTerms;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::mComplexSpaceDim;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::mNumDofsPerNode;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::mNumDofsPerCell;
    using Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>::mNumNodesPerCell;

    using StateScalarType = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType = typename EvaluationType::ConfigScalarType;
    using ResultScalarType = typename EvaluationType::ResultScalarType;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;

    using FunctionBaseType = Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>;

private:
    Plato::Scalar mDensity;

    PenaltyFuncType mPenaltyFunction;
    ProjectionFuncType mProjectionFunction;
    Plato::ApplyPenalty<PenaltyFuncType> mApplyPenalty;
    Plato::ApplyProjection<ProjectionFuncType> mApplyProjection;

    Plato::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

public:
    /**************************************************************************/
    DynamicCompliance(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
   ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "Dynamic Energy"),
        mDensity(aProblemParams.get<Plato::Scalar>("Material Density", 1.0)),
        mProjectionFunction(),
        mPenaltyFunction(aPenaltyParams),
        mApplyPenalty(mPenaltyFunction),
        mApplyProjection(mProjectionFunction),
        mCellStiffness(),
        mCubatureRule(std::make_shared<CubatureType>())
    /**************************************************************************/
    {
        // Create material model and get stiffness
        Plato::ElasticModelFactory<EvaluationType::SpatialDim> tElasticModelFactory(aProblemParams);
        auto tMaterialModel = tElasticModelFactory.create(mSpatialDomain.getMaterialName());
        mCellStiffness = tMaterialModel->getStiffnessMatrix();
    }
    /**************************************************************************/
    DynamicCompliance(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
            FunctionBaseType(aSpatialDomain, aDataMap, "Dynamic Energy"),
            mDensity(1.0),
            mProjectionFunction(),
            mPenaltyFunction(3.0, 0.0),
            mApplyPenalty(mPenaltyFunction),
            mApplyProjection(mProjectionFunction),
            mCellStiffness(),
            mCubatureRule(std::make_shared<CubatureType>())
    /**************************************************************************/
    {
        // Create material model and get stiffness
        Teuchos::ParameterList tParamList;
        tParamList.set < Plato::Scalar > ("Poissons Ratio", 1.0);
        tParamList.set < Plato::Scalar > ("Youngs Modulus", 0.3);
        Plato::IsotropicLinearElasticMaterial<EvaluationType::SpatialDim> tDefaultMaterialModel(tParamList);
        mCellStiffness = tDefaultMaterialModel.getStiffnessMatrix();
    }

    /**************************************************************************/
    virtual ~DynamicCompliance()
    {
    }
    /**************************************************************************/

    /*************************************************************************
     * Evaluate f(u,z)=\frac{1}{2}u^{T}(K(z) - \omega^2 M(z))u, where u denotes
     * states, z denotes controls, K denotes the stiffness matrix and M denotes
     * the mass matrix.
     **************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        using StrainScalarType =
        typename Plato::fad_type_t<Plato::SimplexStructuralDynamics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

        // Elastic forces functors
        Plato::ComplexElasticEnergy<mNumVoigtTerms> tComputeElasticEnergy;
        Plato::ComputeGradientWorkset<EvaluationType::SpatialDim> tComputeGradientWorkset;
        Plato::ComplexStrain<EvaluationType::SpatialDim, mNumDofsPerNode> tComputeVoigtStrain;
        Plato::ComplexLinearStress<EvaluationType::SpatialDim, mNumVoigtTerms> tComputeVoigtStress(mCellStiffness);

        // Inertial forces functors
        Plato::StateValues tComputeStateValues;
        Plato::ComplexInertialEnergy<EvaluationType::SpatialDim> tComputeInertialEnergy(aTimeStep, mDensity);

        // Elastic forces containers
        auto tNumCells = aControl.extent(0);
        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("CellWeight", tNumCells);
        Plato::ScalarArray3DT<StrainScalarType> tCellStrain("CellStrain", tNumCells, mComplexSpaceDim, mNumVoigtTerms);
        Plato::ScalarArray3DT<ResultScalarType> tCellStress("CellStress", tNumCells, mComplexSpaceDim, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigScalarType> tCellGradient("Gradient", tNumCells, mNumNodesPerCell, EvaluationType::SpatialDim);

        // Inertial forces containers
        Plato::ScalarVectorT<ResultScalarType> tElasticEnergy("ElasticEnergy", tNumCells);
        Plato::ScalarVectorT<ResultScalarType> tInertialEnergy("InertialEnergy", tNumCells);
        Plato::ScalarMultiVectorT<StateScalarType> tStateValues("StateValues", tNumCells, mNumDofsPerNode);

        auto & tApplyPenalty = mApplyPenalty;
        auto & tApplyProjection = mApplyProjection;
        auto & tPenaltyFunction = mPenaltyFunction;
        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for("Dynamic Compliance Calculation", Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
        {
            // Internal forces contribution
            tComputeGradientWorkset(aCellOrdinal, tCellGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;
            tComputeVoigtStrain(aCellOrdinal, aState, tCellGradient, tCellStrain);
            tComputeVoigtStress(aCellOrdinal, tCellStrain, tCellStress);

            // Apply penalty to internal forces
            ControlScalarType tCellDensity = tApplyProjection(aCellOrdinal, aControl);
            tApplyPenalty(aCellOrdinal, tCellDensity, tCellStress);
            tComputeElasticEnergy(aCellOrdinal, tCellStress, tCellStrain, tElasticEnergy);
            tElasticEnergy(aCellOrdinal) *= tCellVolume(aCellOrdinal);

            // Inertial forces contribution
            tComputeStateValues(aCellOrdinal, tBasisFunctions, aState, tStateValues);
            tComputeInertialEnergy(aCellOrdinal, tCellVolume, tStateValues, tInertialEnergy);
            ControlScalarType tPenaltyValue = tPenaltyFunction(tCellDensity);
            tInertialEnergy(aCellOrdinal) *= tPenaltyValue;

            // Add inertial forces contribution
            aResult(aCellOrdinal) = static_cast<Plato::Scalar>(0.5) *
                ( tElasticEnergy(aCellOrdinal) + tInertialEnergy(aCellOrdinal) );
        });
    }
};
// class DynamicCompliance

}//namespace Plato

#endif /* DYNAMICCOMPLIANCE_HPP_ */
