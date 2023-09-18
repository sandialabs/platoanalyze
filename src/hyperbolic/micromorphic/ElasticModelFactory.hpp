#pragma once

#include "material/MaterialModel.hpp"
#include "material/MaterialModelFactory.hpp"
#include "hyperbolic/micromorphic/CubicLinearElasticMaterial.hpp"

#include <Teuchos_RCP.hpp>

namespace Plato::Hyperbolic::Micromorphic
{

template<Plato::OrdinalType SpatialDim>
class ElasticModelFactory : public MaterialModelFactory<SpatialDim>
{
public:
    ElasticModelFactory(const Teuchos::ParameterList& aParamList) :
    MaterialModelFactory<SpatialDim>(aParamList)
    {}

protected:
    Teuchos::RCP<Plato::MaterialModel<SpatialDim>>
    constructFromSublist(const Teuchos::ParameterList& aParamList) override
    {
        if(aParamList.isSublist("Cubic Micromorphic Linear Elastic"))
        {
            return Teuchos::rcp(new Plato::Hyperbolic::Micromorphic::CubicLinearElasticMaterial<SpatialDim>(aParamList.sublist("Cubic Micromorphic Linear Elastic")));
        }
        return Teuchos::RCP<Plato::MaterialModel<SpatialDim>>(nullptr);
    }

};

}