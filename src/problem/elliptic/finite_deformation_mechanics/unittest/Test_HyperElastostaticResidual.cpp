#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>

#include "core_types/PlatoTypes.hpp"
#include "domain/SpatialModel.hpp"
#include "element/MechanicsElement.hpp"
#include "element/Tet4.hpp"
#include "element/Tri3.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "problem/elliptic/VectorFunction.hpp"
#include "problem/elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::elliptic::finite_deformation_mechanics::unittest
{
namespace
{
Teuchos::ParameterList create_param_list()
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");
    tParameterList.set("PDE Constraint", "Elliptic");
    tParameterList.set("Physics", "Finite Deformation Mechanics");
    tParameterList.sublist("Elliptic").sublist("Penalty Function").set("Exponent", 1.0);
    tParameterList.sublist("Elliptic").sublist("Penalty Function").set("Minimum Value", 1e-16);
    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Element Block", "body");
    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Material Model", "Pudding");
    tParameterList.sublist("Material Models")
        .sublist("Pudding")
        .sublist("Neo Hookean Hyperelastic")
        .set("Bulk Modulus", 0.5);
    tParameterList.sublist("Material Models")
        .sublist("Pudding")
        .sublist("Neo Hookean Hyperelastic")
        .set("Shear Modulus", 0.375);
    return tParameterList;
}

template <typename ElementType>
Plato::ScalarVector compute_residual_over_mesh(const Plato::Mesh& aMesh,
                                               const Plato::ScalarVector& aState,
                                               const Plato::ScalarVector& aControl)
{
    Teuchos::ParameterList tParamList = create_param_list();

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(aMesh, tParamList, tDataMap);

    Plato::Elliptic::VectorFunction<FiniteDeformationMechanics<typename ElementType::TopoElementType>> tPDE(
        tSpatialModel, tDataMap, tParamList, tParamList.get<std::string>("PDE Constraint"));

    return tPDE.value(aState, aControl);
}
}  // namespace

TEUCHOS_UNIT_TEST(HyperElastostaticResidual, ZeroDisplacementGivesZeroResidual)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();

    const auto tNumDofs = ElementType::mNumSpatialDims * tNumNodes;
    const Plato::ScalarVector tState("zero state", tNumDofs);

    const Plato::ScalarVector tControl("control of ones", tNumNodes);
    Plato::blas1::fill(1.0, tControl);

    const auto tResidual = compute_residual_over_mesh<ElementType>(tMesh, tState, tControl);
    const auto tResidualHost = Plato::TestHelpers::get(tResidual);
    for (Plato::OrdinalType tDofOrdinal = 0; tDofOrdinal < tNumDofs; tDofOrdinal++)
    {
        TEST_ASSERT(tResidualHost(tDofOrdinal) < 1e-16);
    }
}

TEUCHOS_UNIT_TEST(HyperElastostaticResidual, UniaxialDisplacementGivesExpectedResidual)
{
    // 2 Element Tri mesh
    //
    // 1 o ----- o 3 --> u = 1.0
    //   |      / |
    //   |     /  |
    //   |    /   |
    //   |   /    |
    //   |  /     |
    // 0 o -------o 2 --> u = 1.0
    //
    // Connectivity:
    // [0, 2, 3]
    // [0, 3, 1]

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();
    const auto tNumDofs = ElementType::mNumSpatialDims * tNumNodes;

    // Make uniaxial displacement field with extension of 1.0
    std::vector<Plato::Scalar> tDispVals{0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    const auto tState = Plato::TestHelpers::create_device_view(tDispVals);

    std::vector<Plato::Scalar> tGoldResidual{-0.30561759842764435, -0.25688240157235565, -0.30561759842764435,
                                             0.25688240157235565,  0.30561759842764435,  -0.25688240157235565,
                                             0.30561759842764435,  0.25688240157235565};

    // control of 1
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(1.0, tControl);

        const auto tResidual = compute_residual_over_mesh<ElementType>(tMesh, tState, tControl);
        const auto tResidualHost = Plato::TestHelpers::get(tResidual);

        for (Plato::OrdinalType tDofOrdinal = 0; tDofOrdinal < tNumDofs; tDofOrdinal++)
        {
            TEST_FLOATING_EQUALITY(tResidualHost(tDofOrdinal), tGoldResidual[tDofOrdinal], 1e-14);
        }
    }

    // control of 0.5
    {
        constexpr Plato::Scalar tControlValue{0.5};
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(tControlValue, tControl);

        const auto tResidual = compute_residual_over_mesh<ElementType>(tMesh, tState, tControl);
        const auto tResidualHost = Plato::TestHelpers::get(tResidual);

        for (Plato::OrdinalType tDofOrdinal = 0; tDofOrdinal < tNumDofs; tDofOrdinal++)
        {
            TEST_FLOATING_EQUALITY(tResidualHost(tDofOrdinal), tControlValue * tGoldResidual[tDofOrdinal], 1e-14);
        }
    }
}
}  // namespace plato::elliptic::finite_deformation_mechanics::unittest
