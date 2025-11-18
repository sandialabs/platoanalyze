#include "domain/SpatialModel.hpp"

#include <Teuchos_ParameterList.hpp>

#include "linear_algebra/PlatoMathTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMask.hpp"
#include "mesh/PlatoMesh.hpp"
#include "parsing/ParseTools.hpp"

namespace Plato
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
    const auto tBlockName = SpatialDomain::elementBlockName(aDomainParams).value_or(std::string{"UNKNOWN"});
    const auto tErrorMessage = "Element Block in the input file with name " + tBlockName +
                               " has no matching block in the exodus mesh.\nBlock names in mesh: \n";
    const auto tBlockNamesInMesh = block_names_as_string(aMesh);
    return tErrorMessage + tBlockNamesInMesh;
}
}  // namespace

SpatialDomain::SpatialDomain(Plato::Mesh aMesh, Plato::DataMap& aDataMap, std::string aName)
    : Mesh(aMesh), mDataMap(aDataMap), mSpatialDomainName(std::move(aName))
{
}

SpatialDomain::SpatialDomain(Plato::Mesh aMesh,
                             Plato::DataMap& aDataMap,
                             const Teuchos::ParameterList& aInputParams,
                             std::string aName)
    : Mesh(aMesh), mDataMap(aDataMap), mSpatialDomainName(std::move(aName))
{
    this->initialize(aInputParams);
}

auto SpatialDomain::elementBlockExistsInMesh(const Plato::Mesh& aMesh, const Teuchos::ParameterList& aInputParams)
    -> bool
{
    if (const auto tElementBlockName = SpatialDomain::elementBlockName(aInputParams))
    {
        const auto tElementBlocksInMesh = aMesh->GetElementBlockNames();
        return std::find(tElementBlocksInMesh.cbegin(), tElementBlocksInMesh.cend(), tElementBlockName.value()) !=
               tElementBlocksInMesh.cend();
    }
    return false;
}

auto SpatialDomain::elementBlockName(const Teuchos::ParameterList& aInputParams) -> std::optional<std::string>
{
    constexpr auto tElementBlockTag = "Element Block";
    if (aInputParams.isType<std::string>(tElementBlockTag))
    {
        return aInputParams.get<std::string>(tElementBlockTag);
    }
    return std::nullopt;
}

void SpatialDomain::removeMask() { Kokkos::deep_copy(mMaskedElemLids, mTotalElemLids); }

void SpatialDomain::setMaskLocalElemIDs(const std::string& aBlockName)
{
    auto tElemLids = Mesh->GetLocalElementIDs(aBlockName);
    auto tNumElems = tElemLids.size();
    mTotalElemLids = Plato::OrdinalVector("element list", tNumElems);
    mMaskedElemLids = Plato::OrdinalVector("masked element list", tNumElems);

    auto tTotalElemLids = mTotalElemLids;
    Kokkos::parallel_for(
        "get element ids", Kokkos::RangePolicy<>(0, tNumElems), KOKKOS_LAMBDA(const Plato::OrdinalType& aCellOrdinal) {
            tTotalElemLids(aCellOrdinal) = tElemLids[aCellOrdinal];
        });
    Kokkos::deep_copy(mMaskedElemLids, mTotalElemLids);
}

void SpatialDomain::initialize(const Teuchos::ParameterList& aInputParams)
{
    if (aInputParams.isType<std::string>("Element Block"))
    {
        mElementBlockName = aInputParams.get<std::string>("Element Block");
        this->cellOrdinals(mElementBlockName);
    }
    else
    {
        ANALYZE_THROWERR("Parsing new Domain. Required keyword 'Element Block' not found");
    }

    if (aInputParams.isType<std::string>("Material Model"))
    {
        mMaterialModelName = aInputParams.get<std::string>("Material Model");
    }
    else
    {
        ANALYZE_THROWERR("Parsing new Domain. Required keyword 'Material Model' not found");
    }
    if (aInputParams.isType<bool>("Fixed Control"))
    {
        mIsFixedBlock = aInputParams.get<bool>("Fixed Control");
    }

    this->setMaskLocalElemIDs(mElementBlockName);

    parseUniformCartesianBasis(aInputParams);
    parseVaryingCartesianBasis(aInputParams);
}

void SpatialDomain::parseUniformCartesianBasis(const Teuchos::ParameterList& aParamList)
{
    if (aParamList.isSublist("Basis"))
    {
        if (Mesh->NumDimensions() == 3)
        {
            Plato::ParseTools::getBasis(aParamList, mUniformCartesianBasis);
        }
        else if (Mesh->NumDimensions() == 2)
        {
            Plato::Matrix<2, 2> tBasis;
            Plato::ParseTools::getBasis(aParamList, tBasis);
            setUniformCartesianBasis(tBasis);
        }
        else if (Mesh->NumDimensions() == 1)
        {
            Plato::Matrix<1, 1> tBasis;
            Plato::ParseTools::getBasis(aParamList, tBasis);
            setUniformCartesianBasis(tBasis);
        }
        mHasUniformBasis = true;
    }
    else
    {
        mHasUniformBasis = false;
    }
}

void SpatialDomain::parseVaryingCartesianBasis(const Teuchos::ParameterList& aParamList)
{
    if (aParamList.isType<std::string>("Basis Field"))
    {
        auto tBasisFieldName = aParamList.get<std::string>("Basis Field");
        auto tBasisField = mDataMap.scalarArray3Ds[tBasisFieldName];
        mHasVaryingBasis = true;

        auto tBasisDim = tBasisField.extent(1);
        auto tNumCells = this->numCells();
        auto tCellOrds = this->cellOrdinals();
        Kokkos::resize(mVaryingCartesianBasis, tNumCells, tBasisDim, tBasisDim);

        auto& tVaryingCartesianBasis = mVaryingCartesianBasis;
        Kokkos::parallel_for(
            "get basis", Kokkos::RangePolicy<>(0, tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType& aCellOrdinal) {
                auto iCellOrdinal = tCellOrds(aCellOrdinal);
                for (decltype(tBasisDim) iDim = 0; iDim < tBasisDim; iDim++)
                {
                    for (decltype(tBasisDim) jDim = 0; jDim < tBasisDim; jDim++)
                    {
                        tVaryingCartesianBasis(aCellOrdinal, iDim, jDim) = tBasisField(iCellOrdinal, iDim, jDim);
                    }
                }
            });
    }
    else
    {
        mHasVaryingBasis = false;
    }
}

SpatialModel::SpatialModel(Plato::Mesh aMesh) : Mesh(aMesh), mHasContact(false), mUpdateGraphForContact(aMesh) {}

SpatialModel::SpatialModel(Plato::Mesh aMesh, const Teuchos::ParameterList& aInputParams, Plato::DataMap& aDataMap)
    : Mesh(aMesh), mHasContact(false), mUpdateGraphForContact(aMesh)
{
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
            if (SpatialDomain::elementBlockExistsInMesh(aMesh, tDomainParams))
            {
                Domains.emplace_back(aMesh, aDataMap, tDomainParams, tMyName);
            }
            else if (!ignoreMissingElementBlocks(tModelParams))
            {
                ANALYZE_THROWERR(error_message_for_mismatched_blocks(aMesh, tDomainParams));
            }
        }
    }
    else
    {
        ANALYZE_THROWERR("Parsing 'Plato Problem'. Required 'Spatial Model' parameter list not found");
    }
}

auto SpatialModel::ignoreMissingElementBlocks(const Teuchos::ParameterList& aParameterList) -> bool
{
    constexpr auto tParameterName = "Ignore Missing Element Blocks";
    if (aParameterList.isParameter(tParameterName))
    {
        return aParameterList.get<bool>(tParameterName);
    }
    constexpr auto tDefaultValue = false;
    return tDefaultValue;
}

void SpatialModel::append(Plato::SpatialDomain& aDomain) { Domains.push_back(aDomain); }

void SpatialModel::addContact(std::vector<Plato::Contact::ContactPair> aPairs)
{
    if (!mHasContact)
    {
        mContactPairs = aPairs;

        auto tNumNodes = Plato::Contact::count_total_child_nodes(aPairs);
        Plato::OrdinalVector tChildNodes("", tNumNodes);
        Plato::OrdinalVector tParentElements("", tNumNodes);
        Plato::Contact::populate_full_contact_arrays(aPairs, tChildNodes, tParentElements);
        Plato::Contact::check_for_repeated_child_nodes(tChildNodes, Mesh->NumNodes());
        mUpdateGraphForContact.createNodeNodeGraph(tChildNodes, tParentElements);

        mHasContact = true;
    }
}

void SpatialModel::NodeNodeGraph(Plato::OrdinalVector& aOffsetMap, Plato::OrdinalVector& aNodeOrds) const
{
    if (mHasContact)
        mUpdateGraphForContact.NodeNodeGraph(aOffsetMap, aNodeOrds);
    else
        Mesh->NodeNodeGraph(aOffsetMap, aNodeOrds);
}

void SpatialModel::NodeNodeGraphTranspose(Plato::OrdinalVector& aOffsetMap, Plato::OrdinalVector& aNodeOrds) const
{
    if (mHasContact)
        mUpdateGraphForContact.NodeNodeGraphTranspose(aOffsetMap, aNodeOrds);
    else
        Mesh->NodeNodeGraph(aOffsetMap, aNodeOrds);
}

}  // namespace Plato
