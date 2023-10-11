#ifndef PLATO_EXPRESSION_TMKINETICS_HPP
#define PLATO_EXPRESSION_TMKINETICS_HPP

#include "AbstractTMKinetics.hpp"
#include "ExpressionEvaluator.hpp"
#include "InterpolateFromNodal.hpp"

#include "material/TensorConstant.hpp"

namespace Plato
{

/******************************************************************************/
/*! Expression Thermomechanics Kinetics functor.

    given a strain, temperature gradient, and temperature, compute the stress and flux
*/
/******************************************************************************/
template<typename EvaluationType, typename ElementType>
class ExpressionTMKinetics :
    public Plato::AbstractTMKinetics<EvaluationType, ElementType>
{
protected:
    using StateT  = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
    using KinematicsScalarType = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

    using ElementType::mNumVoigtTerms;
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumSpatialDims;

    Plato::TensorConstant<mNumSpatialDims> mThermalExpansivityConstant;
    Plato::TensorConstant<mNumSpatialDims> mThermalConductivityConstant;

    Plato::Scalar mRefTemperature;
    const Plato::Scalar mScaling;
    const Plato::Scalar mScaling2;
    Plato::VoigtMap<mNumSpatialDims> cVoigtMap;

    std::shared_ptr<Plato::Rank4Field<EvaluationType>> mElasticStiffnessField;
    ControlScalarType mControlValue;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model
    **********************************************************************************/
    ExpressionTMKinetics(
      const Teuchos::RCP<Plato::MaterialModel<mNumSpatialDims>>   aMaterialModel,
      const Plato::SpatialDomain                                & aSpatialDomain,
      const Plato::DataMap                                      & aDataMap
    ) :
      AbstractTMKinetics<EvaluationType, ElementType>(aMaterialModel, aSpatialDomain, aDataMap),
      mRefTemperature(aMaterialModel->getScalarConstant("Reference Temperature")),
      mScaling(aMaterialModel->getScalarConstant("Temperature Scaling")),
      mScaling2(mScaling*mScaling)
    {
        mThermalExpansivityConstant = aMaterialModel->getTensorConstant("Thermal Expansivity");
        mThermalConductivityConstant = aMaterialModel->getTensorConstant("Thermal Conductivity");

        mElasticStiffnessField = aMaterialModel->template getRank4Field<EvaluationType>("Elastic Stiffness Expression");

        mControlValue = -1.0;
        if(aMaterialModel->scalarConstantExists("Density"))
        {
            mControlValue = aMaterialModel->getScalarConstant("Density");
        }
    }

    void 
    setLocalControl(const Plato::ScalarMultiVectorT <ControlScalarType> &aControl,
                               Plato::ScalarMultiVectorT<ControlScalarType> &aLocalControl) const
    {
        // This code allows for the user to specify a global density value for all nodes when 
        // running a forward problem (when mControlValue != -1.0). This is set with a "Density" entry in 
        // the input deck (see parsing of this in MaterialModel constructor). Typically, though, the passed in 
        // control will just be used. 
        if(mControlValue != -1.0)
        {
            auto tControlValue = mControlValue;
            Kokkos::parallel_for("Compute local control", Kokkos::RangePolicy<>(0,aControl.extent(0)), KOKKOS_LAMBDA(Plato::OrdinalType i)
            {
                for(Plato::OrdinalType j=0; j<aControl.extent(1); j++)
                {
                    aLocalControl(i,j) = tControlValue;
                }
            });
        }
        else
        {
            Kokkos::parallel_for("Compute local control", Kokkos::RangePolicy<>(0,aControl.extent(0)), KOKKOS_LAMBDA(Plato::OrdinalType i)
            {
                for(Plato::OrdinalType j=0; j<aControl.extent(1); j++)
                {
                    aLocalControl(i,j) = aControl(i,j);
                }
            });
        }
    }

    void
    computeThermalStrainStressAndFlux(
        Plato::OrdinalType                            const & aNumCells,
        Plato::ScalarMultiVectorT<StateT>             const & aTemperature,
        Plato::ScalarMultiVectorT<ControlScalarType>  const & aLocalControl,
        Plato::ScalarArray3DT<KinematicsScalarType>   const & aStrain,
        Plato::ScalarArray3DT<KineticsScalarType>     const & aStress,
        Plato::ScalarArray3DT<KineticsScalarType>     const & aFlux,
        Plato::ScalarArray3DT<KinematicsScalarType>   const & aTGrad
    ) const
    {
        auto tScaling = mScaling;
        auto tScaling2 = mScaling2;
        auto tRefTemperature = mRefTemperature;
        auto& tThermalExpansivityConstant = mThermalExpansivityConstant;
        auto& tThermalConductivityConstant = mThermalConductivityConstant;
        auto& tVoigtMap = cVoigtMap;

        auto tCubWeights = ElementType::getCubWeights();
        auto tNumPoints = tCubWeights.size();

        auto tStiffness = (*mElasticStiffnessField)(aLocalControl);

        Kokkos::parallel_for("compute element kinematics", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {aNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            StateT tTemperature = aTemperature(iCellOrdinal, iGpOrdinal);

            // compute thermal strain
            //
            StateT tstrain[mNumVoigtTerms] = {0};
            for( int iDim=0; iDim<mNumSpatialDims; iDim++ ){
                tstrain[iDim] = tScaling * tThermalExpansivityConstant(tVoigtMap.I[iDim], tVoigtMap.J[iDim])
                            * (tTemperature - tRefTemperature);
            }

            // compute stress
            //
            for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
                aStress(iCellOrdinal, iGpOrdinal, iVoigt) = 0.0;
                for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
                    aStress(iCellOrdinal, iGpOrdinal, iVoigt)
                      += (aStrain(iCellOrdinal, iGpOrdinal, jVoigt)-tstrain[jVoigt])*tStiffness(iCellOrdinal, iGpOrdinal, iVoigt, jVoigt);
                }
            }

            // compute flux
            //
            for( int iDim=0; iDim<EvaluationType::SpatialDim; iDim++){
                aFlux(iCellOrdinal, iGpOrdinal, iDim) = 0.0;
                for( int jDim=0; jDim<mNumSpatialDims; jDim++){
                    aFlux(iCellOrdinal, iGpOrdinal, iDim) += tScaling2 * aTGrad(iCellOrdinal, iGpOrdinal, jDim)*tThermalConductivityConstant(iDim, jDim);
                }
            }
        });
    }

    /***********************************************************************************
     * \brief Compute stress and thermal flux from strain, temperature, and temperature gradient
     * \param [in] aStrain infinitesimal strain tensor
     * \param [in] aTGrad temperature gradient
     * \param [in] aTemperature temperature
     * \param [out] aStress Cauchy stress tensor
     * \param [out] aFlux thermal flux vector
     **********************************************************************************/
    void
    compute(
        Plato::ScalarArray3DT<KineticsScalarType>    const & aStress,
        Plato::ScalarArray3DT<KineticsScalarType>    const & aFlux,
        Plato::ScalarArray3DT<KinematicsScalarType>  const & aStrain,
        Plato::ScalarArray3DT<KinematicsScalarType>  const & aTGrad,
        Plato::ScalarMultiVectorT<StateT>            const & aTemperature,
        Plato::ScalarMultiVectorT<ControlScalarType> const & aControl
    ) const
    {
        const Plato::OrdinalType tNumCells = aStrain.extent(0);

        Plato::ScalarMultiVectorT<ControlScalarType> tLocalControl("Local Control", aControl.extent(0), aControl.extent(1));

        // Set local control to user-defined value if requested.
        setLocalControl(aControl, tLocalControl);

        computeThermalStrainStressAndFlux(tNumCells, aTemperature, tLocalControl, aStrain, aStress, aFlux, aTGrad);
    }
};// class ExpressionTMKinetics
}// namespace Plato
#endif
