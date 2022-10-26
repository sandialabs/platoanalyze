#pragma once

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "SpatialModel.hpp"
#include <Teuchos_ParameterList.hpp>


namespace Plato
{

class ContactPair
{
public:
    ContactPair
    (const Teuchos::ParameterList            & aParams,
     Plato::Mesh                               aMesh,
     const std::vector<Plato::SpatialDomain> & aDomains);

    Plato::OrdinalVector getParentDomainCellMap
    (const std::string                       & aDomainName,
     const std::vector<Plato::SpatialDomain> & aDomains);

    Plato::OrdinalVector fillParentElements
    (const Teuchos::Array<Plato::Scalar> & aGap,
     const Plato::OrdinalVector          & aDomain);
    
    Plato::OrdinalVectorT<const Plato::OrdinalType> childNodesA() {return mChildNodesA;}
    Plato::OrdinalVectorT<const Plato::OrdinalType> childNodesB() {return mChildNodesB;}

    Plato::OrdinalVector parentElementsA() {return mParentElementsA;}
    Plato::OrdinalVector parentElementsB() {return mParentElementsB;}

private:
    Plato::OrdinalVectorT<const Plato::OrdinalType> mChildNodesA;
    Plato::OrdinalVectorT<const Plato::OrdinalType> mChildNodesB;

    Plato::OrdinalVector mParentElementsA;
    Plato::OrdinalVector mParentElementsB;
};

}
// namespace Plato
