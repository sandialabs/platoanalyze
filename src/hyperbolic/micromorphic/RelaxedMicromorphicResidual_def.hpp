#pragma once

#include "hyperbolic/micromorphic/RelaxedMicromorphicResidual_decl.hpp"

#include "hyperbolic/micromorphic/ElasticModelFactory.hpp"
#include "hyperbolic/micromorphic/InertiaModelFactory.hpp"

#include "ToMap.hpp"
#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"

#include "GradientMatrix.hpp"
#include "CellVolume.hpp"

#include "hyperbolic/EvaluationTypes.hpp"

#include "hyperbolic/micromorphic/Kinematics.hpp"
#include "hyperbolic/micromorphic/MicromorphicKineticsFactory.hpp"
#include "hyperbolic/micromorphic/FullStressDivergence.hpp"
#include "hyperbolic/micromorphic/ProjectStressToNode.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "ProjectToNode.hpp"
#include "InterpolateFromNodal.hpp"

#include "BLAS1.hpp"

namespace Plato::Hyperbolic::Micromorphic
{

    template<typename EvaluationType, typename IndicatorFunctionType>
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::RelaxedMicromorphicResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyMassWeighting   (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr)
    {
        if(mNumSpatialDims == 1)
        {
           this->mDofNames = std::vector<std::string>({"Displacement X", "Micro Distortion XX"});
           this->mDofDotNames = std::vector<std::string>({"Velocity X", "Micro Velocity XX"});
           this->mDofDotDotNames = std::vector<std::string>({"Acceleration X", "Micro Acceleration XX"});
        }
        if(mNumSpatialDims == 2)
        {
           this->mDofNames = std::vector<std::string>({"Displacement X", "Displacement Y",
           "Micro Distortion XX", "Micro Distortion YX",
           "Micro Distortion XY", "Micro Distortion YY"});
           this->mDofDotNames = std::vector<std::string>({"Velocity X", "Velocity Y",
           "Micro Velocity XX", "Micro Velocity YX",
           "Micro Velocity XY", "Micro Velocity YY"});
           this->mDofDotDotNames = std::vector<std::string>({"Acceleration X", "Acceleration Y",
           "Micro Acceleration XX", "Micro Acceleration YX",
           "Micro Acceleration XY", "Micro Acceleration YY"});
        }
        if(mNumSpatialDims == 3)
        {
           this->mDofNames = std::vector<std::string>({"Displacement X", "Displacement Y", "Displacement Z",
           "Micro Distortion XX", "Micro Distortion YX", "Micro Distortion ZX",
           "Micro Distortion XY", "Micro Distortion YY", "Micro Distortion ZY",
           "Micro Distortion XZ", "Micro Distortion YZ", "Micro Distortion ZZ"});
           this->mDofDotNames = std::vector<std::string>({"Velocity X", "Velocity Y", "Velocity Z",
           "Micro Velocity XX", "Micro Velocity YX", "Micro Velocity ZX",
           "Micro Velocity XY", "Micro Velocity YY", "Micro Velocity ZY",
           "Micro Velocity XZ", "Micro Velocity YZ", "Micro Velocity ZZ"});
           this->mDofDotDotNames = std::vector<std::string>({"Acceleration X", "Acceleration Y", "Acceleration Z",
           "Micro Acceleration XX", "Micro Acceleration YX", "Micro Acceleration ZX",
           "Micro Acceleration XY", "Micro Acceleration YY", "Micro Acceleration ZY",
           "Micro Acceleration XZ", "Micro Acceleration YZ", "Micro Acceleration ZZ"});
        }

        this->checkTimeIntegrator(aProblemParams.sublist("Time Integration"));

        Plato::Hyperbolic::Micromorphic::ElasticModelFactory<mNumSpatialDims> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        Plato::Hyperbolic::Micromorphic::InertiaModelFactory<mNumSpatialDims> tInertiaModelFactory(aProblemParams);
        mInertiaModel = tInertiaModelFactory.create(aSpatialDomain.getMaterialName());

        mRayleighDamping = false;

        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, ElementType>>
                         (aProblemParams.sublist("Body Loads"));
        }

        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<ElementType>>
                             (aProblemParams.sublist("Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Hyperbolic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    checkTimeIntegrator(Teuchos::ParameterList & aIntegratorParams)
    {
        if (aIntegratorParams.isType<bool>("A-Form"))
        {
            auto tAForm = aIntegratorParams.get<bool>("A-Form");
            auto tBeta = aIntegratorParams.get<double>("Newmark Beta");
            if (tAForm == false)
            {
                ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark A-Form must be specified for micromorphic mechanics")
            }
            else if (tBeta != 0.0)
            {
                ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark explicit (beta=0, gamma=0.5) must be specified for micromorphic mechanics")
            }
        }
        else
        {
            ANALYZE_THROWERR("In RelaxedMicromorphicResidual constructor: Newmark A-Form must be specified for micromorphic mechanics")
        }
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Scalar
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    getMaxEigenvalue(const Plato::ScalarArray3D & aConfig) const
    {
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVector tCellVolume("cell weight", tNumCells);

        Plato::ComputeCellVolume<ElementType> tComputeVolume;

        auto tCubPoints = ElementType::getCubPoints();
        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        Kokkos::parallel_for("compute cell volume", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            Plato::Scalar tVolume;
            auto tCubPoint = tCubPoints(iGpOrdinal);
            tComputeVolume(iCellOrdinal, tCubPoint, aConfig, tVolume);
            Kokkos::atomic_add(&tCellVolume(iCellOrdinal), tVolume);
        });

        Plato::Scalar tMinVolume;
        Plato::blas1::min(tCellVolume, tMinVolume);
        Plato::Scalar tLength = pow(tMinVolume, 1.0/mNumSpatialDims);

        auto tMassDensity = mInertiaModel->getScalarConstant("Mass Density");

        Plato::Scalar tSoundSpeed = 0.0;
        if (mMaterialModel->hasRank4VoigtConstant("Ce"))
        {
            const auto tStiffnessMatrixCe = mMaterialModel->getRank4VoigtConstant("Ce");
            tSoundSpeed = sqrt(tStiffnessMatrixCe(0,0)/tMassDensity);
        }
        else if (mMaterialModel->hasRank4Field("Ce"))
        {
            const auto tStiffnessFieldCe = mMaterialModel->template getRank4Field<Plato::Hyperbolic::ResidualTypes<ElementType>>("Ce");
            Plato::ScalarMultiVectorT<Plato::Scalar> tControl("Control Workset", tNumCells, ElementType::mNumNodesPerCell);
            Plato::blas1::fill(1.0, tControl);
            const auto tStiffnessMatrixCe = (*tStiffnessFieldCe)(tControl);
            tSoundSpeed = sqrt(tStiffnessMatrixCe(0,0,0,0)/tMassDensity);
        }

        return 2.0*tSoundSpeed/tLength;
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    Plato::Solutions 
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
      return aSolutions;
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    evaluate(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep,
              Plato::Scalar aCurrentTime
    ) const
    {
        if ( mRayleighDamping )
        {
             evaluateWithDamping(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult, aTimeStep, aCurrentTime);
        }
        else
        {
             evaluateWithoutDamping(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult, aTimeStep, aCurrentTime);
        }
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    evaluateWithoutDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep,
              Plato::Scalar aCurrentTime
    ) const
    {
      using StrainScalarType =
          typename Plato::fad_type_t<ElementType, StateScalarType, ConfigScalarType>;

      using InertiaStrainScalarType =
          typename Plato::fad_type_t<ElementType, StateDotDotScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientMatrix<ElementType> computeGradient;
      Plato::Hyperbolic::Micromorphic::Kinematics<ElementType> computeKinematics;
      Plato::Hyperbolic::Micromorphic::MicromorphicKineticsFactory<EvaluationType, ElementType> tKineticsFactory;
      auto computeKinetics = tKineticsFactory.create(mMaterialModel);
      auto computeInertiaKinetics = tKineticsFactory.create(mInertiaModel);
      Plato::Hyperbolic::Micromorphic::FullStressDivergence<ElementType> computeFullStressDivergence;
      Plato::Hyperbolic::Micromorphic::ProjectStressToNode<ElementType, mNumSpatialDims> computeStressForMicromorphicResidual;
      Plato::InertialContent<ElementType> computeInertialContent(mInertiaModel);
      Plato::ProjectToNode<ElementType, mNumSpatialDims> projectInertialContent;
      Plato::InterpolateFromNodal<ElementType, mNumDofsPerNode, /*offset=*/0, mNumSpatialDims> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType> 
        tCellSymDisplacementGradient("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tCellSymCauchyStress("stress",tNumCells,mNumVoigtTerms);

      auto tCubPoints = ElementType::getCubPoints();
      auto tCubWeights = ElementType::getCubWeights();
      auto tNumPoints = tCubWeights.size();

      constexpr auto cNumSkwTerms = ElementType::mNumSkwTerms;
      constexpr auto cNumVoigtTerms = ElementType::mNumVoigtTerms;
      constexpr auto cNumSpatialDims = ElementType::mNumSpatialDims;
      constexpr auto cNumNodesPerCell = ElementType::mNumNodesPerCell;

      Plato::ScalarArray4DT<ConfigScalarType> tGradient("shape function gradients",tNumCells,tNumPoints,ElementType::mNumNodesPerCell,ElementType::mNumSpatialDims);
      Plato::ScalarMultiVectorT<ConfigScalarType> tVolume("volume jacobians",tNumCells,tNumPoints);

      Plato::ScalarArray3DT<StrainScalarType> tSymDisplacementGradient("sym displacement gradient",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<StrainScalarType> tSkwDisplacementGradient("skw displacement gradient",tNumCells,tNumPoints,ElementType::mNumSkwTerms);
      Plato::ScalarArray3DT<StateScalarType> tSymMicroDistortionTensor("sym micro distortion tensor",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<StateScalarType> tSkwMicroDistortionTensor("skw micro distortion tensor",tNumCells,tNumPoints,ElementType::mNumSkwTerms);

      Plato::ScalarArray3DT<InertiaStrainScalarType> tSymGradientMicroInertia("sym gradient micro inertia",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<InertiaStrainScalarType> tSkwGradientMicroInertia("skw gradient micro inertia",tNumCells,tNumPoints,ElementType::mNumSkwTerms);
      Plato::ScalarArray3DT<StateDotDotScalarType> tSymFreeMicroInertia      ("sym free micro inertia",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<StateDotDotScalarType> tSkwFreeMicroInertia      ("skw free micro inertia",tNumCells,tNumPoints,ElementType::mNumSkwTerms);

      Plato::ScalarArray3DT<ResultScalarType> tSymCauchyStress("sym cauchy stress",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<ResultScalarType> tSkwCauchyStress("skw cauchy stress",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<ResultScalarType> tSymMicroStress ("sym micro stress",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);

      Plato::ScalarArray3DT<ResultScalarType> tSymGradientInertiaStress("sym gradient inertia stress",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<ResultScalarType> tSkwGradientInertiaStress("skw gradient inertia stress",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<ResultScalarType> tSymFreeInertiaStress    ("sym free inertia stress",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);
      Plato::ScalarArray3DT<ResultScalarType> tSkwFreeInertiaStress    ("skw free inertia stress",tNumCells,tNumPoints,ElementType::mNumVoigtTerms);

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyMassWeighting = mApplyMassWeighting;

      Kokkos::parallel_for("compute kinematics", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          auto tCubPoint = tCubPoints(iGpOrdinal);
          computeGradient(iCellOrdinal,iGpOrdinal,tCubPoint,aConfig,tGradient,tVolume);

          auto tBasisValues = ElementType::basisValues(tCubPoint);

          computeKinematics(iCellOrdinal,iGpOrdinal,
                            tSymDisplacementGradient,tSkwDisplacementGradient,
                            tSymMicroDistortionTensor,tSkwMicroDistortionTensor,
                            aState,tBasisValues,tGradient);

          computeKinematics(iCellOrdinal,iGpOrdinal,
                            tSymGradientMicroInertia,tSkwGradientMicroInertia,
                            tSymFreeMicroInertia,tSkwFreeMicroInertia,
                            aStateDotDot,tBasisValues,tGradient);
      });

      (*computeKinetics)(tSymCauchyStress,tSkwCauchyStress,tSymMicroStress,
                         tSymDisplacementGradient,tSkwDisplacementGradient,
                         tSymMicroDistortionTensor,tSkwMicroDistortionTensor,
                         aControl);

      (*computeInertiaKinetics)(tSymGradientInertiaStress,tSkwGradientInertiaStress,
                                tSymFreeInertiaStress,tSkwFreeInertiaStress,
                                tSymGradientMicroInertia,tSkwGradientMicroInertia,
                                tSymFreeMicroInertia,tSkwFreeMicroInertia,
                                aControl);

      Kokkos::parallel_for("compute residual", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
      {
          auto tCubPoint = tCubPoints(iGpOrdinal);
          auto tBasisValues = ElementType::basisValues(tCubPoint);

          applyStressWeighting(iCellOrdinal,iGpOrdinal,aControl,tBasisValues,tSymCauchyStress);
          applyStressWeighting(iCellOrdinal,iGpOrdinal,aControl,tBasisValues,tSkwCauchyStress);
          applyStressWeighting(iCellOrdinal,iGpOrdinal,aControl,tBasisValues,tSymMicroStress);
          applyStressWeighting(iCellOrdinal,iGpOrdinal,aControl,tBasisValues,tSymGradientInertiaStress);
          applyStressWeighting(iCellOrdinal,iGpOrdinal,aControl,tBasisValues,tSkwGradientInertiaStress);
          applyStressWeighting(iCellOrdinal,iGpOrdinal,aControl,tBasisValues,tSymFreeInertiaStress);
          applyStressWeighting(iCellOrdinal,iGpOrdinal,aControl,tBasisValues,tSkwFreeInertiaStress);

          tVolume(iCellOrdinal,iGpOrdinal) *= tCubWeights(iGpOrdinal);

          computeFullStressDivergence(iCellOrdinal,iGpOrdinal,aResult,
                                      tSymCauchyStress,tSkwCauchyStress,
                                      tGradient,tVolume);

          computeStressForMicromorphicResidual(iCellOrdinal,iGpOrdinal,aResult,
                                               tSymCauchyStress,tSkwCauchyStress,tSymMicroStress,
                                               tBasisValues,tVolume);

          computeFullStressDivergence(iCellOrdinal,iGpOrdinal,aResult,
                                      tSymGradientInertiaStress,tSkwGradientInertiaStress,
                                      tGradient,tVolume);

          computeStressForMicromorphicResidual(iCellOrdinal,iGpOrdinal,aResult,
                                               tSymFreeInertiaStress,tSkwFreeInertiaStress,
                                               tBasisValues,tVolume);

          Plato::Array<ElementType::mNumSpatialDims, StateDotDotScalarType> tAcceleration(0.0);
          Plato::Array<ElementType::mNumSpatialDims, ResultScalarType>      tInertialContent(0.0);

          interpolateFromNodal(iCellOrdinal,tBasisValues,aStateDotDot,tAcceleration);
          computeInertialContent(tInertialContent,tAcceleration);
          applyMassWeighting(iCellOrdinal,aControl,tBasisValues,tInertialContent);
          projectInertialContent(iCellOrdinal,tVolume(iCellOrdinal,iGpOrdinal),
                                 tBasisValues,tInertialContent,aResult);

          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              Kokkos::atomic_add(&tCellSymDisplacementGradient(iCellOrdinal,i), 
                                 tVolume(iCellOrdinal,iGpOrdinal) *
                                 tSymDisplacementGradient(iCellOrdinal,iGpOrdinal,i));
              Kokkos::atomic_add(&tCellSymCauchyStress(iCellOrdinal,i), 
                                 tVolume(iCellOrdinal,iGpOrdinal) *
                                 tSymCauchyStress(iCellOrdinal,iGpOrdinal,i));
          }
          Kokkos::atomic_add(&tCellVolume(iCellOrdinal),tVolume(iCellOrdinal,iGpOrdinal));
      });

      Kokkos::parallel_for("compute cell quantities", Kokkos::RangePolicy<>(0, tNumCells),
      KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal)
      {
          for(int i=0; i<ElementType::mNumVoigtTerms; i++)
          {
              tCellSymDisplacementGradient(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
              tCellSymCauchyStress(iCellOrdinal,i) /= tCellVolume(iCellOrdinal);
          }
      });

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aConfig, aResult, -1.0 );
      }

      if( std::count(mPlotTable.begin(),mPlotTable.end(),"stress") ) { Plato::toMap(mDataMap, tCellSymCauchyStress, "stress", mSpatialDomain); }
      if( std::count(mPlotTable.begin(),mPlotTable.end(),"strain") ) { Plato::toMap(mDataMap, tCellSymDisplacementGradient, "strain", mSpatialDomain); }

    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    evaluateWithDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep,
              Plato::Scalar aCurrentTime
    ) const
    {
        ANALYZE_THROWERR("Relaxed Micromorphic residual does not support damping currently.")
    }

    template<typename EvaluationType, typename IndicatorFunctionType>
    void
    RelaxedMicromorphicResidual<EvaluationType, IndicatorFunctionType>::
    evaluate_boundary(
        const Plato::SpatialModel                                & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep,
              Plato::Scalar aCurrentTime
    ) const
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0, aCurrentTime );
        }
    }

}
