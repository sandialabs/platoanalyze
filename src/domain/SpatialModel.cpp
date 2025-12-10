#include "domain/SpatialModel.hpp"

#include <Teuchos_ParameterList.hpp>

#include "domain/SpatialDomain.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"

namespace plato::domain
{
namespace
{
auto block_names_as_string(const Plato::Mesh& aMesh) -> std::string
{
    const auto tBlockNames = aMesh->GetElementBlockNames();
    auto tBlockNameStream = std::stringstream{};
    std::copy(tBlockNames.cbegin(), tBlockNames.cend(), std::ostream_iterator<std::string>{tBlockNameStream, "\n"});
    return tBlockNameStream.str();
}

auto error_message_for_mismatched_blocks(const Plato::Mesh& aMesh, const Teuchos::ParameterList& aDomainParams)
    -> std::string
{
    const auto tBlockName = detail::element_block_name(aDomainParams).value_or(std::string{"UNKNOWN"});
    const auto tErrorMessage = "Element Block in the input file with name " + tBlockName +
                               " has no matching block in the exodus mesh.\nBlock names in mesh: \n";
    const auto tBlockNamesInMesh = block_names_as_string(aMesh);
    return tErrorMessage + tBlockNamesInMesh;
}
}  // namespace

auto parse_domains(const Teuchos::ParameterList& aInputParams, const Plato::Mesh& aMesh) -> ParsedDomains
{
    ParsedDomains tDomains;
    if (aInputParams.isSublist("Spatial Model"))
    {
        auto tModelParams = aInputParams.sublist("Spatial Model");
        if (!tModelParams.isSublist("Domains"))
        {
            ANALYZE_THROWERR("Parsing 'Spatial Model' parameter list. Required 'Domains' parameter sublist not found");
        }

        auto tDomainsParams = tModelParams.sublist("Domains");
        for (auto tIndex = tDomainsParams.begin(); tIndex != tDomainsParams.end(); ++tIndex)
        {
            const auto& tEntry = tDomainsParams.entry(tIndex);
            const auto& tMyName = tDomainsParams.name(tIndex);

            if (!tEntry.isList())
            {
                ANALYZE_THROWERR(
                    "Parameter in 'Domains' parameter sublist within 'Spatial Model' parameter list not valid.  Expect "
                    "lists only.");
            }

            Teuchos::ParameterList& tDomainParams = tDomainsParams.sublist(tMyName);
            const auto tBlockName = detail::element_block_name(tDomainParams);
            if (detail::element_block_exists_in_mesh(aMesh, tBlockName))
            {
                tDomains.emplace(tMyName, parse_spatial_domain(tDomainParams, aMesh->NumDimensions()));
            }
            else if (!ignore_missing_element_blocks(tModelParams))
            {
                ANALYZE_THROWERR(error_message_for_mismatched_blocks(aMesh, tDomainParams));
            }
        }
    }
    else
    {
        ANALYZE_THROWERR("Parsing 'Plato Problem'. Required 'Spatial Model' parameter list not found");
    }
    return tDomains;
}

auto ignore_missing_element_blocks(const Teuchos::ParameterList& aParameterList) -> bool
{
    constexpr auto tParameterName = "Ignore Missing Element Blocks";
    if (aParameterList.isParameter(tParameterName))
    {
        return aParameterList.get<bool>(tParameterName);
    }
    constexpr auto tDefaultValue = false;
    return tDefaultValue;
}

namespace
{
[[nodiscard]] auto spatial_domains(Plato::Mesh aMesh, const ParsedDomains& aParsedDomains, Plato::DataMap& aDataMap)
    -> std::vector<SpatialDomain>
{
    std::vector<SpatialDomain> tDomains;
    tDomains.reserve(aParsedDomains.size());
    std::ranges::transform(aParsedDomains, std::back_inserter(tDomains),
                           [&aMesh, &aDataMap](const auto& aParsedDomainParameter)
                           {
                               return std::move(SpatialDomain(aMesh, aDataMap, aParsedDomainParameter.second,
                                                              aParsedDomainParameter.first));
                           });
    return tDomains;
}

}  // namespace

SpatialModel::SpatialModel(Plato::Mesh aMesh, const ParsedDomains& aParsedDomains, Plato::DataMap& aDataMap)
    : mMesh(aMesh), mUpdateGraphForContact(aMesh), mDomains(std::move(spatial_domains(aMesh, aParsedDomains, aDataMap)))
{
}

void SpatialModel::addContact(std::vector<Plato::Contact::ContactPair> aPairs)
{
    if (!hasContact())
    {
        mContactPairs = aPairs;

        auto tNumNodes = Plato::Contact::count_total_child_nodes(aPairs);
        Plato::OrdinalVector tChildNodes("", tNumNodes);
        Plato::OrdinalVector tParentElements("", tNumNodes);
        Plato::Contact::populate_full_contact_arrays(aPairs, tChildNodes, tParentElements);
        Plato::Contact::check_for_repeated_child_nodes(tChildNodes, mMesh->NumNodes());
        mUpdateGraphForContact.createNodeNodeGraph(tChildNodes, tParentElements);
    }
}

auto SpatialModel::hasContact() const -> bool { return !mContactPairs.empty(); }

auto SpatialModel::contactPairs() const -> std::vector<Plato::Contact::ContactPair> { return mContactPairs; }

void SpatialModel::nodeNodeGraph(Plato::OrdinalVector& aOffsetMap, Plato::OrdinalVector& aNodeOrds) const
{
    if (hasContact())
        mUpdateGraphForContact.NodeNodeGraph(aOffsetMap, aNodeOrds);
    else
        mMesh->NodeNodeGraph(aOffsetMap, aNodeOrds);
}

void SpatialModel::nodeNodeGraphTranspose(Plato::OrdinalVector& aOffsetMap, Plato::OrdinalVector& aNodeOrds) const
{
    if (hasContact())
        mUpdateGraphForContact.NodeNodeGraphTranspose(aOffsetMap, aNodeOrds);
    else
        mMesh->NodeNodeGraph(aOffsetMap, aNodeOrds);
}

}  // namespace plato::domain
