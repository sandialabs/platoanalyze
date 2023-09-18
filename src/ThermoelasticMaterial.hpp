#pragma once

#include "PlatoStaticsTypes.hpp"
#include "material/MaterialModel.hpp"

#include "material/MaterialModelFactory.hpp"
#include "material/IsotropicStiffnessConstant.hpp"
#include "material/IsotropicStiffnessFunctor.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Base class for Thermoelastic material models
  */
    template<int SpatialDim>
    class ThermoelasticMaterial : public MaterialModel<SpatialDim>
  /******************************************************************************/
  {
  
    public:
      ThermoelasticMaterial(const Teuchos::ParameterList& paramList);

    private:
      void parseElasticStiffness(const Teuchos::ParameterList& paramList);
  };

  /******************************************************************************/
  template<int SpatialDim>
  ThermoelasticMaterial<SpatialDim>::
  ThermoelasticMaterial(const Teuchos::ParameterList& paramList) : MaterialModel<SpatialDim>(paramList)
  /******************************************************************************/
  {
      this->parseElasticStiffness(paramList);
      this->parseTensor("Thermal Expansivity", paramList);
      this->parseTensor("Thermal Conductivity", paramList);

      this->parseScalarConstant("Reference Temperature", paramList, 23.0);
      this->parseScalarConstant("Temperature Scaling", paramList, 1.0);
  }

  /******************************************************************************/
  template<int SpatialDim>
  void ThermoelasticMaterial<SpatialDim>::
  parseElasticStiffness(const Teuchos::ParameterList& aParamList)
  /******************************************************************************/
  {
      if(aParamList.isSublist("Elastic Stiffness"))
      {
          auto tParams = aParamList.sublist("Elastic Stiffness");
          if (tParams.isSublist("Youngs Modulus"))
          {
              this->setRank4VoigtFunctor("Elastic Stiffness", Plato::IsotropicStiffnessFunctor<SpatialDim>(tParams));
          }
          else
          if (tParams.isType<Plato::Scalar>("Youngs Modulus"))
          {
              this->setRank4VoigtConstant("Elastic Stiffness", Plato::IsotropicStiffnessConstant<SpatialDim>(tParams));
          }
          else
          {
              this->parseRank4Voigt("Elastic Stiffness", aParamList);
          }
      }
      else
      if(aParamList.isSublist("Elastic Stiffness Expression"))
      {
          this->parseRank4Field("Elastic Stiffness Expression", aParamList);
      }
  }

  template<int SpatialDim>
  class ThermoelasticModelFactory : public MaterialModelFactory<SpatialDim>
  {
  public:
    ThermoelasticModelFactory(const Teuchos::ParameterList& aParamList) :
    MaterialModelFactory<SpatialDim>(aParamList)
    {}

  protected:
    Teuchos::RCP<Plato::MaterialModel<SpatialDim>>
    constructFromSublist(const Teuchos::ParameterList& aParamList) override
    {
      if( aParamList.isSublist("Thermoelastic") )
      {
        return Teuchos::rcp(new Plato::ThermoelasticMaterial<SpatialDim>(aParamList.sublist("Thermoelastic")));
      }
      else
        ANALYZE_THROWERR("Expected 'Thermoelastic' ParameterList");
    }

  };

} // namespace Plato
