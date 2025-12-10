#include "domain/SpatialDomain.hpp"

#include "parsing/ParseTools.hpp"

namespace plato::domain
{

namespace
{
template <int BasisDim>
[[nodiscard]] auto to_three_by_three(const Plato::Matrix<BasisDim, BasisDim>& aMatrix) -> Plato::Matrix<3, 3>
{
    Plato::Matrix<3, 3> tMatrix;
    for (int i = 0; i < BasisDim; ++i)
    {
        for (int j = 0; j < BasisDim; ++j)
        {
            tMatrix(i, j) = aMatrix(i, j);
        }
    }
    return tMatrix;
}

}  // namespace

auto parse_spatial_domain(const Teuchos::ParameterList& aInputParams, const unsigned int aNumberOfDimensions)
    -> SpatialDomainInputParameters
{
    SpatialDomainInputParameters tParameters;
    tParameters.mNumberOfDimensions = aNumberOfDimensions;
    if (aInputParams.isType<std::string>("Element Block"))
    {
        tParameters.mElementBlockName = aInputParams.get<std::string>("Element Block");
    }
    else
    {
        ANALYZE_THROWERR("Parsing new Domain. Required keyword 'Element Block' not found");
    }

    if (aInputParams.isType<std::string>("Material Model"))
    {
        tParameters.mMaterialModelName = aInputParams.get<std::string>("Material Model");
    }
    else
    {
        ANALYZE_THROWERR("Parsing new Domain. Required keyword 'Material Model' not found");
    }
    if (aInputParams.isType<bool>("Fixed Control"))
    {
        tParameters.mIsFixedBlock = aInputParams.get<bool>("Fixed Control");
    }
    tParameters.mHasUniformBasis = aInputParams.isSublist("Basis");

    if (tParameters.mHasUniformBasis)
    {
        if (aNumberOfDimensions == 3)
        {
            Plato::Matrix<3, 3> tBasis;
            Plato::ParseTools::getBasis(aInputParams, tBasis);
            tParameters.mBasis = tBasis;
        }
        else if (aNumberOfDimensions == 2)
        {
            Plato::Matrix<2, 2> tBasis;
            Plato::ParseTools::getBasis(aInputParams, tBasis);
            tParameters.mBasis = to_three_by_three(tBasis);
        }
        else if (aNumberOfDimensions == 1)
        {
            Plato::Matrix<1, 1> tBasis;
            Plato::ParseTools::getBasis(aInputParams, tBasis);
            tParameters.mBasis = to_three_by_three(tBasis);
        }
    }

    if (aInputParams.isType<std::string>("Basis Field"))
    {
        tParameters.mBasisFieldName = aInputParams.get<std::string>("Basis Field");
    }
    return tParameters;
}

SpatialDomain::SpatialDomain(Plato::Mesh aMesh,
                             Plato::DataMap& aDataMap,
                             const SpatialDomainInputParameters& aSpatialDomainInputParameters,
                             std::string aName)
    : mMesh(aMesh),
      mDataMap(aDataMap),
      mSpatialDomainName(std::move(aName)),
      mElementBlockName(aSpatialDomainInputParameters.mElementBlockName),
      mMaterialModelName(aSpatialDomainInputParameters.mMaterialModelName),
      mIsFixedBlock(aSpatialDomainInputParameters.mIsFixedBlock),
      mHasUniformBasis(aSpatialDomainInputParameters.mHasUniformBasis)
{
    setMaskLocalElemIDs(mElementBlockName);
    if (aSpatialDomainInputParameters.mBasis.has_value())
    {
        setUniformCartesianBasis(aSpatialDomainInputParameters.mBasis.value(),
                                 aSpatialDomainInputParameters.mNumberOfDimensions);
    }

    if (aSpatialDomainInputParameters.mBasisFieldName.has_value())
    {
        mHasVaryingBasis = true;
        varyingCartesianBasis(aSpatialDomainInputParameters.mBasisFieldName.value());
    }
}

void SpatialDomain::setMaskLocalElemIDs(const std::string& aBlockName)
{
    auto tElemLids = mMesh->GetLocalElementIDs(aBlockName);
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

void SpatialDomain::setUniformCartesianBasis(const Plato::Matrix<3, 3>& aBasis, const unsigned int aNumberOfDimensions)
{
    for (int i = 0; i < aNumberOfDimensions; ++i)
    {
        for (int j = 0; j < aNumberOfDimensions; ++j)
        {
            mUniformCartesianBasis(i, j) = aBasis(i, j);
        }
    }
}

void SpatialDomain::varyingCartesianBasis(const std::string& aBasisFieldName)
{
    auto tBasisField = mDataMap.scalarArray3Ds[aBasisFieldName];
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

auto SpatialDomain::domainName() const -> std::string { return mSpatialDomainName; }

auto SpatialDomain::materialName() const -> std::string { return mMaterialModelName; }

auto SpatialDomain::elementBlockName() const -> std::string { return mElementBlockName; }

auto SpatialDomain::fixedBlock() const -> bool { return mIsFixedBlock; }

auto SpatialDomain::numCells() const -> Plato::OrdinalType { return mMaskedElemLids.extent(0); }

auto SpatialDomain::numNodes() const -> Plato::OrdinalType { return mMesh->NumNodes(); }

auto SpatialDomain::cellOrdinals() const -> const Plato::OrdinalVector& { return mMaskedElemLids; }

auto SpatialDomain::hasUniformCartesianBasis() const -> bool { return mHasUniformBasis; }

auto SpatialDomain::hasVaryingCartesianBasis() const -> bool { return mHasVaryingBasis; }

auto SpatialDomain::getVaryingCartesianBasis() const -> Plato::ScalarArray3D { return mVaryingCartesianBasis; }
namespace detail
{
auto element_block_exists_in_mesh(const Plato::Mesh& aMesh, const std::optional<std::string>& aElementBlockName) -> bool
{
    if (aElementBlockName.has_value())
    {
        const auto tElementBlocksInMesh = aMesh->GetElementBlockNames();
        return std::find(tElementBlocksInMesh.cbegin(), tElementBlocksInMesh.cend(), aElementBlockName.value()) !=
               tElementBlocksInMesh.cend();
    }
    return false;
}
auto element_block_name(const Teuchos::ParameterList& aInputParams) -> std::optional<std::string>
{
    constexpr auto tElementBlockTag = "Element Block";
    if (aInputParams.isType<std::string>(tElementBlockTag))
    {
        return aInputParams.get<std::string>(tElementBlockTag);
    }
    return std::nullopt;
}

}  // namespace detail

}  // namespace plato::domain
