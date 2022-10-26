#include "ContactPair.hpp"
#include "Plato_MeshMap.hpp"

#include <string>

namespace Plato
{

    ContactPair::ContactPair
    (const Teuchos::ParameterList            & aParams,
     Plato::Mesh                               aMesh,
     const std::vector<Plato::SpatialDomain> & aDomains)
    {
        if (!aParams.isType<std::string>("Side A Child Sideset"))
            ANALYZE_THROWERR("Side A Child Sideset was not provided in contact pair")

        std::string tSideSetA = aParams.get<std::string>("Side A Child Sideset");
        mChildNodesA = aMesh->GetNodeSetNodes(tSideSetA);

        if (!aParams.isType<std::string>("Side B Child Sideset"))
            ANALYZE_THROWERR("Side B Child Sideset was not provided in contact pair")

        std::string tSideSetB = aParams.get<std::string>("Side B Child Sideset");
        mChildNodesB = aMesh->GetNodeSetNodes(tSideSetB);
        
        // parse initial gap
        if (!aParams.isType<Teuchos::Array<Plato::Scalar>>("Initial Gap"))
            ANALYZE_THROWERR("Initial Gap vector was not provided in contact pair")

        auto tVector = aParams.get<Teuchos::Array<Plato::Scalar>>("Initial Gap");
        Plato::OrdinalType tNumDims = aMesh->NumDimensions();
        if(tVector.size() != tNumDims)
            ANALYZE_THROWERR("Initial Gap vector provided in contact pair has different dimensions than mesh")

        if(tNumDims != 3)
            ANALYZE_THROWERR("Initial Gap vector provided in contact pair must have 3 dimensions. Contact is not implemented for fewer dimensions.")

        // get spatial domains for finding parent elements
        if (!aParams.isType<std::string>("Side B Block"))
            ANALYZE_THROWERR("Side B Block was not provided in contact pair")

        std::string tBlockB = aParams.get<std::string>("Side B Block");
        
        auto tParentDomainCellMap = this->getParentDomainCellMap(tBlockB, aDomains);

        // get child node and mapped coordinates for each side

        // find and store parent elements for each side
    }

    Plato::OrdinalVector ContactPair::getParentDomainCellMap
    (const std::string                       & aDomainName,
     const std::vector<Plato::SpatialDomain> & aDomains)
    {
        Plato::OrdinalVector tDomainCellMap;
        bool tFindName = 0;
        for(auto& tDomain : aDomains)
        {
            auto tName = tDomain.getElementBlockName();
            if( tName == aDomainName )
            {
                tDomainCellMap = tDomain.cellOrdinals();
                tFindName = 1;
            }
        }
        if( tFindName == 0 )
        {
            std::string tMsg = "Block with name " + aDomainName + " provided in contact pair does not correspond to an element block name in spatial model";
            ANALYZE_THROWERR(tMsg)
        }
        return tDomainCellMap;
    }

    Plato::OrdinalVector ContactPair::fillParentElements
    (const Teuchos::Array<Plato::Scalar> & aGap,
     const Plato::OrdinalVector          & aDomain)
    {
        // Plato::Scalar tGapX = tVector[0];
        // Plato::Scalar tGapY = tVector[1];
        // Plato::Scalar tGapZ = tVector[2];
        // Plato::Geometry::findParentElements<ElementType, Plato::Scalar>
        //     (tSpatialModel.Mesh, tDomainCellMap, tChildNodeCoords, tChildNodeMappedCoords, tParentElements);

    }
}
// namespace Plato
