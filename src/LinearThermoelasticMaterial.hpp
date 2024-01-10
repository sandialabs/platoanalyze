#ifndef LINEARTHERMOELASTICMATERIAL_HPP
#define LINEARTHERMOELASTICMATERIAL_HPP

#include "PlatoMathTypes.hpp"
#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato {

/******************************************************************************/
/*!
  \brief Base class for Linear Thermoelastic material models
*/
  template<int SpatialDim>
  class LinearThermoelasticMaterial
/******************************************************************************/
{
  protected:
    static constexpr auto mNumVoigtTerms = (SpatialDim == 3) ? 6 : 
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));
    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3.");

    Plato::Scalar mCellDensity;
    Plato::Scalar mCellSpecificHeat;
    Plato::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    Plato::Array<SpatialDim> mCellThermalExpansionCoef;
    Plato::Matrix<SpatialDim, SpatialDim> mCellThermalConductivity;
    Plato::Scalar mCellReferenceTemperature;

    Plato::Scalar mTemperatureScaling;
    Plato::Scalar mPressureScaling;

  public:
    LinearThermoelasticMaterial(const Teuchos::ParameterList& paramList);
    decltype(mCellDensity)               getMassDensity()          const {return mCellDensity;}
    decltype(mCellSpecificHeat)          getSpecificHeat()         const {return mCellSpecificHeat;}
    decltype(mCellStiffness)             getStiffnessMatrix()      const {return mCellStiffness;}
    decltype(mCellThermalExpansionCoef)  getThermalExpansion()     const {return mCellThermalExpansionCoef;}
    decltype(mCellThermalConductivity)   getThermalConductivity()  const {return mCellThermalConductivity;}
    decltype(mCellReferenceTemperature)  getReferenceTemperature() const {return mCellReferenceTemperature;}
    decltype(mTemperatureScaling)        getTemperatureScaling()   const {return mTemperatureScaling;}
    decltype(mPressureScaling)           getPressureScaling()      const {return mPressureScaling;}
};

/******************************************************************************/
template<int SpatialDim>
LinearThermoelasticMaterial<SpatialDim>::
LinearThermoelasticMaterial(const Teuchos::ParameterList& paramList)
/******************************************************************************/
{
    for(int i=0; i<mNumVoigtTerms; i++)
      for(int j=0; j<mNumVoigtTerms; j++)
        mCellStiffness(i,j) = 0.0;

    for(int i=0; i<SpatialDim; i++)
      mCellThermalExpansionCoef(i) = 0.0;

    for(int i=0; i<SpatialDim; i++)
      for(int j=0; j<SpatialDim; j++)
        mCellThermalConductivity(i,j) = 0.0;

    Plato::Scalar t = paramList.get<Plato::Scalar>("Reference Temperature");
    mCellReferenceTemperature=t;

    if( paramList.isType<Plato::Scalar>("Mass Density") ){
      mCellDensity = paramList.get<Plato::Scalar>("Mass Density");
    } else {
      mCellDensity = 1.0;
    }
    if( paramList.isType<Plato::Scalar>("Specific Heat") ){
      mCellSpecificHeat = paramList.get<Plato::Scalar>("Specific Heat");
    } else {
      mCellSpecificHeat = 1.0;
    }
    if( paramList.isType<Plato::Scalar>("Temperature Scaling") ){
      mTemperatureScaling = paramList.get<Plato::Scalar>("Temperature Scaling");
    } else {
      mTemperatureScaling = 1.0;
    }
    if( paramList.isType<Plato::Scalar>("Pressure Scaling") ){
      mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    } else {
      mPressureScaling = 1.0;
    }
}

/******************************************************************************/
/*!
  \brief Derived class for isotropic linear thermoelastic material model
*/
  template<int SpatialDim>
  class IsotropicLinearThermoelasticMaterial : public LinearThermoelasticMaterial<SpatialDim>
/******************************************************************************/
{
  public:
    IsotropicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList);
    virtual ~IsotropicLinearThermoelasticMaterial(){}
};

/******************************************************************************/
/*!
  \brief Derived class for cubic linear thermoelastic material model
*/
  template<int SpatialDim>
  class CubicLinearThermoelasticMaterial : public LinearThermoelasticMaterial<SpatialDim>
/******************************************************************************/
{
  public:
    CubicLinearThermoelasticMaterial(const Teuchos::ParameterList& paramList);
    virtual ~CubicLinearThermoelasticMaterial(){}
};

/******************************************************************************/
/*!
  \brief Factory for creating material models
*/
  template<int SpatialDim>
  class LinearThermoelasticModelFactory
/******************************************************************************/
{
  public:
    LinearThermoelasticModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
    Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpatialDim>> create(std::string aModelName);
  private:
    const Teuchos::ParameterList& mParamList;
};

/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearThermoelasticMaterial<SpatialDim>>
LinearThermoelasticModelFactory<SpatialDim>::create(std::string aModelName)
/******************************************************************************/
{
    if (!mParamList.isSublist("Material Models"))
    {
        REPORT("'Material Models' list not found! Returning 'nullptr'");
        return Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpatialDim>>(nullptr);
    }
    else
    {
        auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");

        if (!tModelsParamList.isSublist(aModelName))
        {
            std::stringstream ss;
            ss << "Requested a material model ('" << aModelName << "') that isn't defined";
            ANALYZE_THROWERR(ss.str());
        }

        auto tModelParamList = tModelsParamList.sublist(aModelName);
        if( tModelParamList.isSublist("Isotropic Linear Thermoelastic") )
        {
            return Teuchos::rcp(new Plato::IsotropicLinearThermoelasticMaterial<SpatialDim>(tModelParamList.sublist("Isotropic Linear Thermoelastic")));
        }
        else
        if( tModelParamList.isSublist("Cubic Linear Thermoelastic") )
        {
            return Teuchos::rcp(new Plato::CubicLinearThermoelasticMaterial<SpatialDim>(tModelParamList.sublist("Cubic Linear Thermoelastic")));
        }
        return Teuchos::RCP<Plato::LinearThermoelasticMaterial<SpatialDim>>(nullptr);
    }
}

} // namespace Plato

#endif
