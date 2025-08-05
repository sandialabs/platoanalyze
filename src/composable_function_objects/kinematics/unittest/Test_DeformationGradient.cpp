#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include "BLAS1.hpp"
#include "GradientMatrix.hpp"
#include "MechanicsElement.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTestHelpers.hpp"
#include "PlatoTypes.hpp"
#include "Tet4.hpp"
#include "Tri3.hpp"
#include "WorksetBase.hpp"
#include "composable_function_objects/kinematics/DeformationGradient.hpp"

namespace plato::composable_function_objects::kinematics::unittest
{
namespace
{
constexpr Plato::OrdinalType kDeformationGradientSize{3};
constexpr Plato::OrdinalType kPlaneStrainGradientSize{2};

[[nodiscard]] auto full_deformation_gradient_from_plane_strain_displacement_gradient(
    const Plato::Matrix<kPlaneStrainGradientSize, kPlaneStrainGradientSize>& a2DDisplacementGradient)
    -> Plato::Matrix<kDeformationGradientSize, kDeformationGradientSize>
{
    Plato::Matrix<kDeformationGradientSize, kDeformationGradientSize> tReturn(0.0);
    for (Plato::OrdinalType i = 0; i < kPlaneStrainGradientSize; i++)
    {
        for (Plato::OrdinalType j = 0; j < kPlaneStrainGradientSize; j++)
        {
            tReturn(i, j) = a2DDisplacementGradient(i, j);
        }
    }
    return Plato::plus(tReturn, Plato::identity<kDeformationGradientSize>());
}

template <Plato::OrdinalType NumDims>
Plato::ScalarVector create_linear_displacement_field(
    const Plato::Mesh& aMesh, const Plato::Matrix<NumDims, NumDims, Plato::Scalar>& aConstantDisplacementGradient)
{
    const Plato::OrdinalType tNumNodes = aMesh->NumNodes();
    const auto tCoords = aMesh->Coordinates();
    const auto tNumDofs = NumDims * tNumNodes;
    Plato::ScalarVector tDisplacementField("linear displacement", tNumDofs);
    Kokkos::parallel_for(
        "fill linear displacement field", Kokkos::RangePolicy<int>(0, tNumNodes),
        KOKKOS_LAMBDA(Plato::OrdinalType tNodeOrdinal) {
            Plato::Array<NumDims, Plato::Scalar> tNodeCoords(0.0);
            for (Plato::OrdinalType tDofOrdinal = 0; tDofOrdinal < NumDims; tDofOrdinal++)
            {
                tNodeCoords(tDofOrdinal) = tCoords(tNodeOrdinal * NumDims + tDofOrdinal);
            }
            const auto tNodeDisplacement = Plato::times(aConstantDisplacementGradient, tNodeCoords);
            for (Plato::OrdinalType tDofOrdinal = 0; tDofOrdinal < NumDims; tDofOrdinal++)
            {
                tDisplacementField(tNodeOrdinal * NumDims + tDofOrdinal) = tNodeDisplacement(tDofOrdinal);
            }
        });

    return tDisplacementField;
}

template <typename ElementType>
void test_deformation_gradient_against_gold(
    const Plato::Mesh& aMesh,
    const Plato::ScalarVector& aDisplacementField,
    const Plato::Matrix<kDeformationGradientSize, kDeformationGradientSize, Plato::Scalar>& aGoldDeformationGradient,
    Teuchos::FancyOStream& aOutStream,
    bool& aSuccess)
{
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;
    constexpr Plato::OrdinalType tNodesPerCell = ElementType::mNumNodesPerCell;
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;

    const auto tCubPoints = ElementType::getCubPoints();
    auto tCubWeights = ElementType::getCubWeights();
    auto tNumPoints = tCubWeights.size();
    const Plato::OrdinalType tNumCells = aMesh->NumElements();

    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpatialDims);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);

    Plato::WorksetBase<ElementType> tWorksetBase(aMesh);
    tWorksetBase.worksetConfig(tConfigWS);
    tWorksetBase.worksetState(aDisplacementField, tStateWS);

    Plato::ComputeGradientMatrix<ElementType> tComputeGradient;
    DeformationGradient<ElementType> tComputeDeformationGradient;

    Plato::ScalarArray3DT<Plato::Scalar> tDeformationGradients("store deformation gradients", tNumCells, tNumPoints,
                                                               kDeformationGradientSize * kDeformationGradientSize);
    Kokkos::parallel_for(
        "compute on device", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal) {
            auto tCubPoint = tCubPoints(gpOrdinal);

            Plato::Scalar tVolume(0.0);
            Plato::Matrix<tNodesPerCell, tSpatialDims, Plato::Scalar> tShapeGradients(0.0);
            tComputeGradient(cellOrdinal, tCubPoint, tConfigWS, tShapeGradients, tVolume);

            Plato::Matrix<kDeformationGradientSize, kDeformationGradientSize, Plato::Scalar> tDeformationGradient(0.0);
            tComputeDeformationGradient(cellOrdinal, tDeformationGradient, tStateWS, tShapeGradients);

            for (Plato::OrdinalType i = 0; i < kDeformationGradientSize; i++)
            {
                for (Plato::OrdinalType j = 0; j < kDeformationGradientSize; j++)
                {
                    tDeformationGradients(cellOrdinal, gpOrdinal, i * kDeformationGradientSize + j) =
                        tDeformationGradient(i, j);
                }
            }
        });

    const auto tDeformationGradientsHost = Plato::TestHelpers::get(tDeformationGradients);
    for (Plato::OrdinalType tCellOrdinal = 0; tCellOrdinal < tNumCells; tCellOrdinal++)
    {
        for (Plato::OrdinalType tGPOrdinal = 0; tGPOrdinal < tNumPoints; tGPOrdinal++)
        {
            for (Plato::OrdinalType i = 0; i < kDeformationGradientSize; i++)
            {
                for (Plato::OrdinalType j = 0; j < kDeformationGradientSize; j++)
                {
                    TEUCHOS_TEST_FLOATING_EQUALITY(
                        tDeformationGradientsHost(tCellOrdinal, tGPOrdinal, i * kDeformationGradientSize + j),
                        aGoldDeformationGradient(i, j), 1e-14, aOutStream, aSuccess);
                }
            }
        }
    }
}
}  // namespace

TEUCHOS_UNIT_TEST(DeformationGradient, ZeroDisplacementGivesIdentity)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tMeshWidth = 1;

    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    const auto tNumDofs = ElementType::mNumSpatialDims * tMesh->NumNodes();
    Plato::ScalarVector tDisplacementField("zero displacement", tNumDofs);

    const auto tGoldDeformationGradient = Plato::identity<kDeformationGradientSize>();
    test_deformation_gradient_against_gold<ElementType>(tMesh, tDisplacementField, tGoldDeformationGradient, out,
                                                        success);
}

TEUCHOS_UNIT_TEST(DeformationGradient, ConstantDisplacementGivesIdentity)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr Plato::OrdinalType tMeshWidth = 1;

    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    const Plato::OrdinalType tNumCells = tMesh->NumElements();

    const auto tNumDofs = ElementType::mNumSpatialDims * tMesh->NumNodes();
    Plato::ScalarVector tDisplacementField("constant displacement", tNumDofs);
    Plato::blas1::fill(86.0, tDisplacementField);

    const auto tGoldDeformationGradient = Plato::identity<kDeformationGradientSize>();
    test_deformation_gradient_against_gold<ElementType>(tMesh, tDisplacementField, tGoldDeformationGradient, out,
                                                        success);
}

TEUCHOS_UNIT_TEST(DeformationGradient, LinearDisplacementGivesConstant)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    const Plato::Matrix<tSpatialDims, tSpatialDims, Plato::Scalar> tGoldDisplacementGradient{5.9, 2.1, 8.6, 3.8, 7.1,
                                                                                             2.2, 9.3, 2.0, 3.7};
    const auto tDisplacementField = create_linear_displacement_field(tMesh, tGoldDisplacementGradient);

    const auto tGoldDeformationGradient =
        Plato::plus(tGoldDisplacementGradient, Plato::identity<kDeformationGradientSize>());
    test_deformation_gradient_against_gold<ElementType>(tMesh, tDisplacementField, tGoldDeformationGradient, out,
                                                        success);
}

TEUCHOS_UNIT_TEST(DeformationGradient, LinearDisplacementGivesConstant2D)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);

    const Plato::Matrix<tSpatialDims, tSpatialDims, Plato::Scalar> tGoldDisplacementGradient{8.8, 7.1, 8.6, 3.8};
    const auto tDisplacementField = create_linear_displacement_field(tMesh, tGoldDisplacementGradient);

    const auto tGoldDeformationGradient =
        full_deformation_gradient_from_plane_strain_displacement_gradient(tGoldDisplacementGradient);
    test_deformation_gradient_against_gold<ElementType>(tMesh, tDisplacementField, tGoldDeformationGradient, out,
                                                        success);
}

}  // namespace plato::composable_function_objects::kinematics::unittest
