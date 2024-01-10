#ifndef THERMAL_CONTENT_HPP
#define THERMAL_CONTENT_HPP

#include "PlatoStaticsTypes.hpp"
#include "ThermalMassMaterial.hpp"
#include "material/ScalarFunctor.hpp"

namespace Plato
{

/******************************************************************************/
/*! Thermal content functor.
  
    given a temperature value, compute the thermal content
*/
/******************************************************************************/
template<int SpatialDim>
class ThermalContent
{
  private:
    Plato::MaterialModelType mModelType;

    // in case functor is nonlinear
    Plato::ScalarFunctor mMassDensityFunctor;
    Plato::ScalarFunctor mSpecificHeatFunctor;

    // in case functor is linear
    Plato::Scalar mMassDensity;
    Plato::Scalar mSpecificHeat;

  public:
    ThermalContent(const Teuchos::RCP<Plato::MaterialModel<SpatialDim>> aMaterialModel)
    {
        mModelType = aMaterialModel->type();
        if (mModelType == Plato::MaterialModelType::Nonlinear)
        {
            mMassDensityFunctor = aMaterialModel->getScalarFunctor("Mass Density");
            mSpecificHeatFunctor = aMaterialModel->getScalarFunctor("Specific Heat");
        } else
        if (mModelType == Plato::MaterialModelType::Linear)
        {
            mMassDensity = aMaterialModel->getScalarConstant("Mass Density");
            mSpecificHeat = aMaterialModel->getScalarConstant("Specific Heat");
        }
    }

    template<typename TScalarType, typename TRateScalarType, typename TContentScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        TContentScalarType & aContent,
        TRateScalarType      aTemperatureRate,
        TScalarType          aTemperature
    ) const
    {
      // compute thermal content
      //
      if (mModelType == Plato::MaterialModelType::Linear)
      {
          aContent = aTemperatureRate*mMassDensity*mSpecificHeat;
      }
      else
      if (mModelType == Plato::MaterialModelType::Nonlinear)
      {
          TScalarType tMassDensity = mMassDensityFunctor(aTemperature);
          TScalarType tSpecificHeat = mSpecificHeatFunctor(aTemperature);
          aContent = aTemperatureRate*tMassDensity*tSpecificHeat;
      }
    }

    template<typename TScalarType, typename TRateScalarType, typename TContentScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarVectorT<TContentScalarType> tcontent,
                Plato::ScalarVectorT<TRateScalarType> temperature_rate,
                Plato::ScalarVectorT<TScalarType> temperature) const {

      // compute thermal content
      //

      TScalarType cellT = temperature(cellOrdinal);
      TRateScalarType cellTRate = temperature_rate(cellOrdinal);
      if (mModelType == Plato::MaterialModelType::Linear)
      {
          tcontent(cellOrdinal) = cellTRate*mMassDensity*mSpecificHeat;
      } else
      if (mModelType == Plato::MaterialModelType::Nonlinear)
      {
          TScalarType tMassDensity = mMassDensityFunctor(cellT);
          TScalarType tSpecificHeat = mSpecificHeatFunctor(cellT);
          tcontent(cellOrdinal) = cellTRate*tMassDensity*tSpecificHeat;
      }
    }
    template<typename TRateScalarType, typename TContentScalarType>
    KOKKOS_INLINE_FUNCTION void
    operator()(
        TContentScalarType & tcontent,
        TRateScalarType      temperature_rate
    ) const {

      // compute thermal content
      //

      tcontent = temperature_rate*mMassDensity*mSpecificHeat;
    }
};
// class ThermalContent

} // namespace Plato

#endif
