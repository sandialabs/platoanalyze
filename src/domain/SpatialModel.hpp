#ifndef PLATO_DOMAIN_SPATIALMODEL
#define PLATO_DOMAIN_SPATIALMODEL

#include <Teuchos_ParameterList.hpp>

#include "domain/SpatialDomain.hpp"
#include "domain/contact/ContactPair.hpp"
#include "domain/contact/UpdateGraphForContact.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMask.hpp"
#include "mesh/PlatoMesh.hpp"

namespace plato::domain
{

using ParsedDomains = std::map<std::string, SpatialDomainInputParameters>;

/// @brief Take a teuchos input @a aInputParams and the mesh @a aMesh and determine all the parsed domains in the
/// problem set up
[[nodiscard]] auto parse_domains(const Teuchos::ParameterList& aInputParams, const Plato::Mesh& aMesh) -> ParsedDomains;

/// @brief Checks @a aParameterList for the ignore missing elements entry
[[nodiscard]] auto ignore_missing_element_blocks(const Teuchos::ParameterList& aParameterList) -> bool;

/// @brief Spatial models contain the mesh, meshsets, domains, etc that define a discretized geometry.
class SpatialModel
{
   public:
    /// @brief Construct a Spatial model from a mesh @a aMesh, the parsed domains @a aParsedDomains, and an auxilary
    /// data map @a aDataMap
    SpatialModel(Plato::Mesh aMesh, const ParsedDomains& aParsedDomains, Plato::DataMap& aDataMap);

    /// @brief Apply a mask @a aMask to hide some of the spatial model
    template <Plato::OrdinalType SpatialDim>
    void applyMask(std::shared_ptr<Plato::Mask<SpatialDim>> aMask);

    /// @brief Modify the spatial model by adding the contacts defined in @a aPairs
    void addContact(std::vector<Plato::Contact::ContactPair> aPairs);

    /// @brief Modify the graphs using @a aOffsetMap and @a aNodeOrds in the mesh or contact model
    void nodeNodeGraph(Plato::OrdinalVector& aOffsetMap, Plato::OrdinalVector& aNodeOrds) const;

    /// @brief Modify the graph transpose using @a aOffsetMap and @a aNodeOrds in the mesh or contact model
    void nodeNodeGraphTranspose(Plato::OrdinalVector& aOffsetMap, Plato::OrdinalVector& aNodeOrds) const;

    /// @brief Return the contact pairs defined in the spatial model
    [[nodiscard]] auto contactPairs() const -> std::vector<Plato::Contact::ContactPair>;

    /// @brief Return whether the spatial model has contact pairs defined
    [[nodiscard]] auto hasContact() const -> bool;

   public:
    Plato::Mesh mMesh;
    std::vector<SpatialDomain> mDomains;

   private:
    std::vector<Plato::Contact::ContactPair> mContactPairs;
    Plato::Contact::UpdateGraphForContact mUpdateGraphForContact;
};
// class SpatialModel

template <Plato::OrdinalType SpatialDim>
void SpatialModel::applyMask(std::shared_ptr<Plato::Mask<SpatialDim>> aMask)
{
    for (auto& tDomain : mDomains)
    {
        tDomain.applyMask(aMask);
    }
}

}  // namespace plato::domain
#endif
