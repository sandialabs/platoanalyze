#pragma once

#include "material/MaterialModel.hpp"
#include "material/MaterialModelFactory.hpp"
#include "hyperbolic/micromorphic/CubicInertiaMaterial.hpp"

#include <Teuchos_RCP.hpp>

namespace Plato::Hyperbolic::Micromorphic
{

template<Plato::OrdinalType SpatialDim>
class InertiaModelFactory : public MaterialModelFactory<SpatialDim>
{
public:
    InertiaModelFactory(const Teuchos::ParameterList& aParamList) :
    MaterialModelFactory<SpatialDim>(aParamList)
    {}

protected:
    Teuchos::RCP<Plato::MaterialModel<SpatialDim>>
    constructFromSublist(const Teuchos::ParameterList& aParamList) override
    {
        if(aParamList.isSublist("Cubic Micromorphic Inertia"))
        {
            return Teuchos::rcp(new Plato::Hyperbolic::Micromorphic::CubicInertiaMaterial<SpatialDim>(aParamList.sublist("Cubic Micromorphic Inertia")));
        }
        return Teuchos::RCP<Plato::MaterialModel<SpatialDim>>(nullptr);
    }

};

}