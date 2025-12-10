#ifndef PLATO_DOMAIN_SPATIALDOMAIN
#define PLATO_DOMAIN_SPATIALDOMAIN

#include <Teuchos_ParameterList.hpp>
#include <optional>

#include "linear_algebra/PlatoMathTypes.hpp"
#include "mesh/PlatoMask.hpp"
#include "mesh/PlatoMesh.hpp"

namespace plato::domain
{

/// @brief Helper stringly-struct used to create a domain.
struct SpatialDomainInputParameters
{
    std::string mElementBlockName;
    std::string mMaterialModelName;
    bool mIsFixedBlock = false;
    bool mHasUniformBasis = false;
    unsigned int mNumberOfDimensions = 3;
    std::optional<std::string> mBasisFieldName = std::nullopt;
    std::optional<Plato::Matrix<3, 3>> mBasis = std::nullopt;
};

/// @brief Given input @a aInputParams and the working dimensions @a aNumberOfDimensions from the mesh, determine the
/// spatial domain input parameters
[[nodiscard]] auto parse_spatial_domain(const Teuchos::ParameterList& aInputParams,
                                        const unsigned int aNumberOfDimensions) -> SpatialDomainInputParameters;

/// @brief Class for handling domain specific data
class SpatialDomain
{
   public:
    /// @brief Construct a spatial domain with a given name @a aName, from a mesh @a aMesh and a data map @a aDatMap
    /// using specified input in @a aSpatialDomainInputParameters
    SpatialDomain(Plato::Mesh aMesh,
                  Plato::DataMap& aDataMap,
                  const SpatialDomainInputParameters& aSpatialDomainInputParameters,
                  std::string aName);

    /// @brief Return domain name
    [[nodiscard]] auto domainName() const -> std::string;

    /// @brief Return material model name
    [[nodiscard]] auto materialName() const -> std::string;

    /// @brief Return block name of the domain
    [[nodiscard]] auto elementBlockName() const -> std::string;

    /// @brief Return whether this block is fixed
    [[nodiscard]] auto fixedBlock() const -> bool;

    /// @brief Return the number of cells in the domain
    [[nodiscard]] auto numCells() const -> Plato::OrdinalType;

    /// @brief Return number of nodes in the domain
    [[nodiscard]] auto numNodes() const -> Plato::OrdinalType;

    /// @brief return const reference of the cell ordinal vector
    [[nodiscard]] auto cellOrdinals() const -> const Plato::OrdinalVector&;

    /// @brief Apply mask to this Domain. This function removes elements that have a mask value of zero in @a aMask.
    ///        Subsequent calls to numCells() and cellOrdinals() refer to the reduced list.
    ///        Call applyMask(...) to apply a different mask.
    template <Plato::OrdinalType SpatialDim>
    void applyMask(std::shared_ptr<Plato::Mask<SpatialDim>> aMask);

    // The cartesian basis is stored in the 3D matrix, mUniformCartesianBasis,
    // regardless of the actual dimension of the problem.  The accessors below
    // return only the relevant data for the requested dimension.
    template <int BasisDim>
    void getUniformCartesianBasis(Plato::Matrix<BasisDim, BasisDim>& aBasis) const;

    /// @brief Return whether the domain has a uniform cartesian basis
    [[nodiscard]] auto hasUniformCartesianBasis() const -> bool;

    /// @brief Return whether the domain has a varying cartesian basis
    [[nodiscard]] auto hasVaryingCartesianBasis() const -> bool;

    /// @brief Retrieve the varying cartesian basis
    [[nodiscard]] auto getVaryingCartesianBasis() const -> Plato::ScalarArray3D;

   private:
    /// @brief Using @a aBasis and the actual dimensions in the matrix given by @a aNumberOfDimensions, update the
    /// member data
    void setUniformCartesianBasis(const Plato::Matrix<3, 3>& aBasis, const unsigned int aNumberOfDimensions);

    /// @brief Update the varying cartesian basis using @a aBasisFieldName
    void varyingCartesianBasis(const std::string& aBasisFieldName);

    /// @brief Set the local element ids of the mask for @a aBlockName
    void setMaskLocalElemIDs(const std::string& aBlockName);

   public:
    Plato::Mesh mMesh;

   private:
    std::string mElementBlockName;
    std::string mMaterialModelName;
    std::string mSpatialDomainName;
    bool mIsFixedBlock = false;

    Plato::OrdinalVector mTotalElemLids;
    Plato::OrdinalVector mMaskedElemLids;

    Plato::DataMap mDataMap;

    bool mHasUniformBasis = false;
    Plato::Matrix<3, 3> mUniformCartesianBasis;

    bool mHasVaryingBasis = false;
    Plato::ScalarArray3D mVaryingCartesianBasis;
};

namespace detail
{
/// @brief Returns whether @a aElementBlockName exists in the mesh @a aMesh.
[[nodiscard]] auto element_block_exists_in_mesh(const Plato::Mesh& aMesh,
                                                const std::optional<std::string>& aElementBlockName) -> bool;

/// @brief Returns the `Element Block` entry from @a aInputParams.
[[nodiscard]] auto element_block_name(const Teuchos::ParameterList& aInputParams) -> std::optional<std::string>;

}  // namespace detail

template <int BasisDim>
void SpatialDomain::getUniformCartesianBasis(Plato::Matrix<BasisDim, BasisDim>& aBasis) const
{
    for (int i = 0; i < BasisDim; ++i)
    {
        for (int j = 0; j < BasisDim; ++j)
        {
            aBasis(i, j) = mUniformCartesianBasis(i, j);
        }
    }
}

template <Plato::OrdinalType SpatialDim>
void SpatialDomain::applyMask(std::shared_ptr<Plato::Mask<SpatialDim>> aMask)
{
    using OrdinalT = Plato::OrdinalType;

    auto tMask = aMask->cellMask();
    auto tTotalElemLids = mTotalElemLids;
    auto tNumEntries = tTotalElemLids.extent(0);

    // how many non-zeros in the mask?
    Plato::OrdinalType tSum(0);
    Kokkos::parallel_reduce(
        Kokkos::RangePolicy<>(0, tNumEntries),
        KOKKOS_LAMBDA(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType& aUpdate) {
            auto tElemOrdinal = tTotalElemLids(aOrdinal);
            aUpdate += tMask(tElemOrdinal);
        },
        tSum);
    Kokkos::resize(mMaskedElemLids, tSum);

    auto tMaskedElemLids = mMaskedElemLids;

    // create a list of elements with non-zero mask values
    OrdinalT tOffset(0);
    Kokkos::parallel_scan(
        Kokkos::RangePolicy<OrdinalT>(0, tNumEntries),
        KOKKOS_LAMBDA(const OrdinalT& aOrdinal, OrdinalT& aUpdate, const bool& tIsFinal) {
            auto tElemOrdinal = tTotalElemLids(aOrdinal);
            const OrdinalT tVal = tMask(tElemOrdinal);
            if (tIsFinal && tVal)
            {
                tMaskedElemLids(aUpdate) = tElemOrdinal;
            }
            aUpdate += tVal;
        },
        tOffset);
}

}  // namespace plato::domain

#endif
