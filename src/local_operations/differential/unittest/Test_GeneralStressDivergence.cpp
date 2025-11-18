#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include "core_types/PlatoTypes.hpp"
#include "domain/WorksetBase.hpp"
#include "element/MechanicsElement.hpp"
#include "element/Tet4.hpp"
#include "element/Tri3.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "local_operations/differential/GeneralStressDivergence.hpp"
#include "local_operations/differential/GradientMatrix.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::composable_function_objects::shape_function_operations::unittest
{
namespace
{
/// @brief constructs a shape gradient matrix for interpolated differentiation of 2D matrices stored in Voigt form.
/// The shape gradient matrix has the form:
///
/// B = [dN^1/dx, 0,       ... dN^n/dx, 0
///      0,       dN^1/dy, ... 0,       dN^n/dy
///      dN^1/dy, dN^1/dx, ... dN^n/dy, dN^n/dx]
///
/// where N^i is the shape function for node i and n is the number of nodes per element.
template <typename ElementType>
struct VoigtShapeGradientMatrix2D : public ElementType
{
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    KOKKOS_INLINE_FUNCTION auto operator()(const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> aShapeGradients) const
        -> Plato::Matrix<mNumVoigtTerms, mNumDofsPerNode * mNumNodesPerCell>
    {
        Plato::Matrix<mNumVoigtTerms, mNumDofsPerNode * mNumNodesPerCell> tShapeGradMatrix(0.0);
        for (Plato::OrdinalType tNodeOrdinal = 0; tNodeOrdinal < mNumNodesPerCell; tNodeOrdinal++)
        {
            const auto tDofOrdinal = tNodeOrdinal * mNumDofsPerNode;
            tShapeGradMatrix(0, tDofOrdinal) = aShapeGradients(tNodeOrdinal, 0);
            tShapeGradMatrix(1, tDofOrdinal + 1) = aShapeGradients(tNodeOrdinal, 1);

            tShapeGradMatrix(2, tDofOrdinal) = aShapeGradients(tNodeOrdinal, 1);
            tShapeGradMatrix(2, tDofOrdinal + 1) = aShapeGradients(tNodeOrdinal, 0);
        }
        return tShapeGradMatrix;
    }
};

/// @brief constructs a shape gradient matrix for interpolated differentiation of 3D matrices stored in Voigt form.
/// The shape gradient matrix has the form:
///
/// B = [dN^1/dx, 0,       0,       ... dN^n/dx, 0,       0
///      0,       dN^1/dy, 0,       ... 0,       dN^n/dy, 0
///      0,       0,       dN^1/dz, ... 0,       0,       dN^n/dz
///      0,       dN^1/dz, dN^1/dy, ... 0,       dN^n/dz, dN^n/dy
///      dN^1/dz, 0,       dN^1/dx, ... dN^n/dz, 0,       dN^n/dx
///      dN^1/dy, dN^1/dx, 0,       ... dN^n/dy, dN^n/dx, 0]
///
/// where N^i is the shape function for node i and n is the number of nodes per element.
template <typename ElementType>
struct VoigtShapeGradientMatrix3D : public ElementType
{
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;
    using ElementType::mNumVoigtTerms;

    KOKKOS_INLINE_FUNCTION auto operator()(const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> aShapeGradients) const
        -> Plato::Matrix<mNumVoigtTerms, mNumDofsPerNode * mNumNodesPerCell>
    {
        Plato::Matrix<mNumVoigtTerms, mNumDofsPerNode * mNumNodesPerCell> tShapeGradMatrix(0.0);
        for (Plato::OrdinalType tNodeOrdinal = 0; tNodeOrdinal < mNumNodesPerCell; tNodeOrdinal++)
        {
            const auto tDofOrdinal = tNodeOrdinal * mNumDofsPerNode;
            tShapeGradMatrix(0, tDofOrdinal) = aShapeGradients(tNodeOrdinal, 0);
            tShapeGradMatrix(1, tDofOrdinal + 1) = aShapeGradients(tNodeOrdinal, 1);
            tShapeGradMatrix(2, tDofOrdinal + 2) = aShapeGradients(tNodeOrdinal, 2);

            tShapeGradMatrix(3, tDofOrdinal + 1) = aShapeGradients(tNodeOrdinal, 2);
            tShapeGradMatrix(3, tDofOrdinal + 2) = aShapeGradients(tNodeOrdinal, 1);

            tShapeGradMatrix(4, tDofOrdinal) = aShapeGradients(tNodeOrdinal, 2);
            tShapeGradMatrix(4, tDofOrdinal + 2) = aShapeGradients(tNodeOrdinal, 0);

            tShapeGradMatrix(5, tDofOrdinal) = aShapeGradients(tNodeOrdinal, 1);
            tShapeGradMatrix(5, tDofOrdinal + 1) = aShapeGradients(tNodeOrdinal, 0);
        }
        return tShapeGradMatrix;
    }
};

/// @brief constructs a shape gradient matrix for interpolated differentiation of 2D or 3D matrices stored as vectors
/// with full entries
// A 2D stress tensor T stored in full vector form is [T11, T12, T21, T22].
///
/// The shape gradient matrix in 2D (plane strain) has the form:
///
/// B = [dN^1/dx, 0,       ... dN^n/dx, 0
///      dN^1/dy, 0,       ... dN^n/dy, 0
///      0,       dN^1/dx, ... 0,       dN^n/dx
///      0,       dN^1/dy, ... 0,       dN^n/dy]
///
/// where N^i is the shape function for node i and n is the number of nodes per element.
template <typename ElementType>
struct FullShapeGradientMatrix : public ElementType
{
    using ElementType::mNumDofsPerNode;
    using ElementType::mNumNodesPerCell;
    using ElementType::mNumSpatialDims;

    KOKKOS_INLINE_FUNCTION auto operator()(const Plato::Matrix<mNumNodesPerCell, mNumSpatialDims> aShapeGradients) const
        -> Plato::Matrix<mNumSpatialDims * mNumSpatialDims, mNumDofsPerNode * mNumNodesPerCell>
    {
        Plato::Matrix<mNumSpatialDims * mNumSpatialDims, mNumDofsPerNode * mNumNodesPerCell> tShapeGradMatrix(0.0);
        for (Plato::OrdinalType tNodeOrdinal = 0; tNodeOrdinal < mNumNodesPerCell; tNodeOrdinal++)
        {
            for (Plato::OrdinalType tDofOrdinal = 0; tDofOrdinal < mNumDofsPerNode; tDofOrdinal++)
            {
                for (Plato::OrdinalType tDimOrdinal = 0; tDimOrdinal < mNumSpatialDims; tDimOrdinal++)
                {
                    tShapeGradMatrix(tDofOrdinal * mNumDofsPerNode + tDimOrdinal,
                                     tNodeOrdinal * mNumDofsPerNode + tDofOrdinal) =
                        aShapeGradients(tNodeOrdinal, tDimOrdinal);
                }
            }
        }
        return tShapeGradMatrix;
    }
};

template <typename ElementType>
struct DummyVoigtStress
{
    KOKKOS_INLINE_FUNCTION auto operator()() const -> Plato::Array<ElementType::mNumVoigtTerms>
    {
        Plato::Array<ElementType::mNumVoigtTerms> tDummyStress(0.0);
        for (Plato::OrdinalType tStressOrdinal = 0; tStressOrdinal < ElementType::mNumVoigtTerms; tStressOrdinal++)
        {
            tDummyStress(tStressOrdinal) = 1.0 / (tStressOrdinal + 1.0);
        }
        return tDummyStress;
    }
};

template <typename ElementType>
struct DummyFullStress
{
    KOKKOS_INLINE_FUNCTION auto operator()() const
        -> Plato::Matrix<ElementType::mNumSpatialDims, ElementType::mNumSpatialDims>
    {
        Plato::Matrix<ElementType::mNumSpatialDims, ElementType::mNumSpatialDims> tDummyStress(0.0);
        for (Plato::OrdinalType i = 0; i < ElementType::mNumSpatialDims; i++)
        {
            for (Plato::OrdinalType j = 0; j < ElementType::mNumSpatialDims; j++)
            {
                tDummyStress(i, j) = 1.0 / (j + 1.0);
            }
        }
        return tDummyStress;
    }
};

/// @brief tests the correct implementation of GeneralStressDivergence operation by comparing to stress divergence
/// values computed using shape gradient matrices B, i.e. \nabla . \sigma = B^T [\sigma].
template <typename ElementType,
          template <typename> typename ConstructShapeGradientMatrixType,
          template <typename> typename ConstructDummyStressType>
void test_stress_divergence_operator_against_matrix_multiplication_approach(const Plato::Mesh& aMesh,
                                                                            Teuchos::FancyOStream& aOutStream,
                                                                            bool& aSuccess)
{
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    const Plato::OrdinalType tNumCells = aMesh->NumElements();

    const auto tCubPoints = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints = tCubWeights.size();

    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpatialDims);
    Plato::WorksetBase<ElementType> tWorksetBase(aMesh);
    tWorksetBase.worksetConfig(tConfigWS);

    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    GeneralStressDivergence<ElementType> tComputeStressDivergence;
    ConstructShapeGradientMatrixType<ElementType> tConstructShapeGradientMatrix;
    ConstructDummyStressType<ElementType> tConstructDummyStress;

    Plato::ScalarMultiVectorT<Plato::Scalar> tStoreDivergence("store divergence", tNumCells, tDofsPerCell);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStoreMatrixDivergence("store matrix multiplication divergence", tNumCells,
                                                                    tDofsPerCell);

    Kokkos::parallel_for(
        "compute on device", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal) {
            auto tCubPoint = tCubPoints(gpOrdinal);

            Plato::Scalar tVolume(0.0);
            Plato::Matrix<tNodesPerCell, tSpatialDims> tShapeGradients(0.0);
            tComputeGradient(cellOrdinal, tCubPoint, tConfigWS, tShapeGradients, tVolume);

            const auto tArbitraryScaleFactor = cellOrdinal * (2.0 + gpOrdinal);
            const auto tDummyStress = Plato::times(tArbitraryScaleFactor, tConstructDummyStress());

            tComputeStressDivergence(cellOrdinal, tStoreDivergence, tDummyStress, tShapeGradients, tVolume);

            const auto tShapeGradMatrix = tConstructShapeGradientMatrix(tShapeGradients);
            const auto tMatrixMultiplicationDivergence =
                Plato::times(Plato::transpose(tShapeGradMatrix), Plato::flatten(tDummyStress));
            for (Plato::OrdinalType tDofOrdinal = 0; tDofOrdinal < tDofsPerCell; tDofOrdinal++)
            {
                Kokkos::atomic_add(
                    &tStoreMatrixDivergence(cellOrdinal, tDofOrdinal),
                    tVolume * tMatrixMultiplicationDivergence(
                                  tDofOrdinal));  // scale by volume to be consistent with GeneralStressDivergence
            }
        });

    const auto tDivergenceHost = Plato::TestHelpers::get(tStoreDivergence);
    const auto tMatrixDivergenceHost = Plato::TestHelpers::get(tStoreMatrixDivergence);
    for (Plato::OrdinalType tCellOrdinal = 0; tCellOrdinal < tNumCells; tCellOrdinal++)
    {
        for (Plato::OrdinalType tDofOrdinal = 0; tDofOrdinal < tDofsPerCell; tDofOrdinal++)
        {
            TEUCHOS_TEST_FLOATING_EQUALITY(tDivergenceHost(tCellOrdinal, tDofOrdinal),
                                           tMatrixDivergenceHost(tCellOrdinal, tDofOrdinal), 1e-14, aOutStream,
                                           aSuccess);
        }
    }
}
}  // namespace

TEUCHOS_UNIT_TEST(GeneralStressDivergence, VoigtFormOperatorMatchesMatrixMultiplication2D)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    test_stress_divergence_operator_against_matrix_multiplication_approach<ElementType, VoigtShapeGradientMatrix2D,
                                                                           DummyVoigtStress>(tMesh, out, success);
}

TEUCHOS_UNIT_TEST(GeneralStressDivergence, VoigtFormOperatorMatchesMatrixMultiplication3D)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    test_stress_divergence_operator_against_matrix_multiplication_approach<ElementType, VoigtShapeGradientMatrix3D,
                                                                           DummyVoigtStress>(tMesh, out, success);
}

TEUCHOS_UNIT_TEST(GeneralStressDivergence, FullFormOperatorMatchesMatrixMultiplication2D)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    test_stress_divergence_operator_against_matrix_multiplication_approach<ElementType, FullShapeGradientMatrix,
                                                                           DummyFullStress>(tMesh, out, success);
}

TEUCHOS_UNIT_TEST(GeneralStressDivergence, FullFormOperatorMatchesMatrixMultiplication3D)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    test_stress_divergence_operator_against_matrix_multiplication_approach<ElementType, FullShapeGradientMatrix,
                                                                           DummyFullStress>(tMesh, out, success);
}
}  // namespace plato::composable_function_objects::shape_function_operations::unittest
