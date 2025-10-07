#ifndef PLATOMATHTESTHELPERS_HPP_
#define PLATOMATHTESTHELPERS_HPP_

#include <Kokkos_StdAlgorithms.hpp>
#include <Teuchos_RCP.hpp>
#include <vector>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{
namespace TestHelpers
{

/// @brief A return type for comparision tests providing if the comparison was `true` or `false`, and also the reason if
/// `false`.
using TestComparisonResult = std::pair<bool, std::string>;

template <typename DataType>
void set_view_from_vector(Plato::ScalarVectorT<DataType> aView, const std::vector<DataType> &aVector);

/// @brief Returns a vector with a copy of the data in @a aView.
template <typename DataType>
[[nodiscard]] auto to_vector(Plato::ScalarVectorT<DataType> aView) -> std::vector<DataType>;

void set_matrix_data(Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
                     const std::vector<Plato::OrdinalType> &aRowMap,
                     const std::vector<Plato::OrdinalType> &aColMap,
                     const std::vector<Plato::Scalar> &aValues);

/// @brief Creates a square CRS matrix with number of rows and columns @a aNumberOfRows.
[[nodiscard]] auto square_crs_matrix(const OrdinalType aNumberOfRows,
                                     const std::vector<Plato::OrdinalType> &aRowMap,
                                     const std::vector<Plato::OrdinalType> &aColMap,
                                     const std::vector<Plato::Scalar> &aValues) -> Plato::CrsMatrixType;

void from_full(Teuchos::RCP<Plato::CrsMatrixType> aOutMatrix, const std::vector<std::vector<Plato::Scalar>> &aInMatrix);

void row_sum(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix, Plato::ScalarVector &aOutRowSum);

void inverse_multiply(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix, Plato::ScalarVector &aInDiagonal);

void slow_dumb_row_summed_inverse_multiply(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
                                           Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo);

void matrix_minus_equals_matrix(Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
                                const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo);

void matrix_minus_equals_matrix(Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
                                const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
                                Plato::OrdinalType aOffset);

void slow_dumb_matrix_minus_matrix(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
                                   const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
                                   int aOffset = -1);

void slow_dumb_matrix_matrix_multiply(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixOne,
                                      const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixTwo,
                                      Teuchos::RCP<Plato::CrsMatrixType> &aOutMatrix);

template <typename DataType>
bool is_same(const Plato::ScalarVectorT<DataType> &aView, const std::vector<DataType> &aVec);

template <typename DataType>
bool is_same(const Plato::ScalarVectorT<DataType> &aViewA, const Plato::ScalarVectorT<DataType> &aViewB);

/// @brief Compares all entries of @a aViewA and @a aViewB and returns `true` if all are within @a aTolerance using
/// absolute difference.
template <typename DataType>
[[nodiscard]] auto is_near(const Plato::ScalarVectorT<DataType> &aViewA,
                           const Plato::ScalarVectorT<DataType> &aViewB,
                           double aTolerance) -> TestComparisonResult;

/// @brief Compares all entries of @a aViewA and @a aViewB and returns `true` if all are within @a aTolerance using
/// absolute difference.
template <typename DataType>
[[nodiscard]] auto is_near(const Plato::ScalarVectorT<DataType> &aView,
                           const std::vector<DataType> &aVec,
                           double aTolerance) -> TestComparisonResult;

bool is_sequential(const Plato::ScalarVectorT<Plato::OrdinalType> &aRowMap,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMap);

bool is_equivalent(const Plato::ScalarVectorT<Plato::OrdinalType> &aRowMap,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMapA,
                   const Plato::ScalarVectorT<Plato::Scalar> &aValuesA,
                   const Plato::ScalarVectorT<Plato::OrdinalType> &aColMapB,
                   const Plato::ScalarVectorT<Plato::Scalar> &aValuesB,
                   Plato::Scalar tolerance = 1.0e-14);

bool is_zero(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrix);

bool is_same(const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixA,
             const Teuchos::RCP<Plato::CrsMatrixType> &aInMatrixB);

template <typename DataType>
void print_view(const Plato::ScalarVectorT<DataType> &aView);

// Implementations
template <typename DataType>
void set_view_from_vector(Plato::ScalarVectorT<DataType> aView, const std::vector<DataType> &aVector)
{
    Kokkos::View<const DataType *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(aVector.data(),
                                                                                         aVector.size());
    Kokkos::deep_copy(aView, tHostView);
}

template <typename DataType>
auto to_vector(const Plato::ScalarVectorT<DataType> aView) -> std::vector<DataType>
{
    auto tMirrorView = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), aView);
    auto tVector = std::vector<DataType>{};
    tVector.reserve(tMirrorView.size());
    std::copy(Kokkos::Experimental::begin(tMirrorView), Kokkos::Experimental::end(tMirrorView),
              std::back_inserter(tVector));
    return tVector;
}

template <typename DataType>
bool is_same(const Plato::ScalarVectorT<DataType> &aViewA, const Plato::ScalarVectorT<DataType> &aViewB)
{
    if (aViewA.extent(0) != aViewB.extent(0)) return false;

    auto tViewA_host = Kokkos::create_mirror(aViewA);
    Kokkos::deep_copy(tViewA_host, aViewA);
    auto tViewB_host = Kokkos::create_mirror(aViewB);
    Kokkos::deep_copy(tViewB_host, aViewB);
    for (unsigned int i = 0; i < aViewA.extent(0); ++i)
    {
        if (tViewA_host(i) != tViewB_host(i)) return false;
    }
    return true;
}

template <typename DataType>
bool is_same(const Plato::ScalarVectorT<DataType> &aView, const std::vector<DataType> &aVec)
{
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);
    for (unsigned int i = 0; i < aVec.size(); ++i)
    {
        if (tView_host(i) != aVec[i])
        {
            return false;
        }
    }
    return true;
}

template <typename DataType>
auto is_near(const Plato::ScalarVectorT<DataType> &aViewA,
             const Plato::ScalarVectorT<DataType> &aViewB,
             double aTolerance) -> TestComparisonResult
{
    if (aViewA.extent(0) != aViewB.extent(0))
    {
        return {false, "Sizes don't match"};
    }

    auto tMirrorViewA = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), aViewA);
    auto tMirrorViewB = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultHostExecutionSpace(), aViewB);
    for (auto tIndex = 0U; tIndex < aViewA.extent(0); ++tIndex)
    {
        if (std::fabs(tMirrorViewA(tIndex) - tMirrorViewB(tIndex)) > aTolerance)
        {
            return {false, "Entries don't match at index " + std::to_string(tIndex) + ": " +
                               std::to_string(tMirrorViewA(tIndex)) + " vs. " + std::to_string(tMirrorViewB(tIndex))};
        }
    }
    return {true, ""};
}

template <typename DataType>
auto is_near(const Plato::ScalarVectorT<DataType> &aView,
             const std::vector<DataType> &aVector,
             double aTolerance) -> TestComparisonResult
{
    auto tViewB = Plato::ScalarVectorT<DataType>{"View b", aView.size()};
    set_view_from_vector(tViewB, aVector);
    return is_near(aView, tViewB, aTolerance);
}

template <typename DataType>
void print_view(const Plato::ScalarVectorT<DataType> &aView)
{
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);
    std::cout << '\n';
    for (unsigned int i = 0; i < aView.extent(0); ++i)
    {
        std::cout << tView_host(i) << '\n';
    }
}
}  // namespace TestHelpers
}  // namespace Plato

#endif
