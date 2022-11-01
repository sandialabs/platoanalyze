#pragma once

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "SpatialModel.hpp"

#include <Teuchos_ParameterList.hpp>
#include <string>

namespace Plato
{

struct ContactPair
{
    Plato::OrdinalVectorT<const Plato::OrdinalType> childNodesA;
    Plato::OrdinalVectorT<const Plato::OrdinalType> childNodesB;

    std::string parentBlockA;
    std::string parentBlockB;

    Teuchos::Array<Plato::Scalar> initialGap;
};

ContactPair parseContactPair
(const Teuchos::ParameterList            & aParams,
 Plato::Mesh                               aMesh);

Plato::SpatialDomain getDomain
(const std::string                       & aDomainName,
 const std::vector<Plato::SpatialDomain> & aDomains);

Plato::ScalarMultiVector computeNodeLocations
(Plato::Mesh                                             aMesh,
 const Plato::OrdinalVectorT<const Plato::OrdinalType> & aNodes);

Plato::ScalarMultiVector mapNodeLocations
(const Plato::ScalarMultiVector      & aLocations,
 const Teuchos::Array<Plato::Scalar> & aTranslation,
 Plato::Scalar                         aScale = 1.0);

}
// namespace Plato
