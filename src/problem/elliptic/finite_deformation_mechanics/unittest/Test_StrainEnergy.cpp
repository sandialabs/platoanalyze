#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <vector>

#include "core_types/PlatoTypes.hpp"
#include "domain/Solutions.hpp"
#include "domain/SpatialModel.hpp"
#include "element/MechanicsElement.hpp"
#include "element/Tet4.hpp"
#include "element/Tri3.hpp"
#include "linear_algebra/BLAS1.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "problem/elliptic/PhysicsScalarFunction.hpp"
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

    tParameterList.sublist("Criteria").sublist("Strain Energy").set("Type", "Scalar Function");
    tParameterList.sublist("Criteria").sublist("Strain Energy").set("Scalar Function Type", "Strain Energy");
    tParameterList.sublist("Criteria").sublist("Strain Energy").sublist("Penalty Function").set("Type", "SIMP");
    tParameterList.sublist("Criteria").sublist("Strain Energy").sublist("Penalty Function").set("Exponent", 1.0);
    tParameterList.sublist("Criteria").sublist("Strain Energy").sublist("Penalty Function").set("Minimum Value", 1e-16);

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
Plato::Scalar compute_criterion_over_mesh(const Plato::Mesh& aMesh,
                                          const std::vector<Plato::Scalar>& aStateVector,
                                          const Plato::ScalarVector& aControl)
{
    Teuchos::ParameterList tParamList = create_param_list();
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(aMesh, tParamList, tDataMap);
    const std::string tCriterionName = "Strain Energy";
    const Plato::Elliptic::PhysicsScalarFunction<FiniteDeformationMechanics<typename ElementType::TopoElementType>>
        tCriterion(tSpatialModel, tDataMap, tParamList, tCriterionName);

    const auto tSolution = Plato::TestHelpers::single_step_solutions_from_vector(aStateVector);

    return tCriterion.value(tSolution, aControl);
}
}  // namespace

TEUCHOS_UNIT_TEST(StrainEnergy, ZeroDisplacementGivesZeroResidual)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();
    const auto tNumDofs = ElementType::mNumSpatialDims * tNumNodes;

    const std::vector<Plato::Scalar> tStateVector(tNumDofs, 0.0);

    const Plato::ScalarVector tControl("control of ones", tNumNodes);
    Plato::blas1::fill(1.0, tControl);

    const auto tValue = compute_criterion_over_mesh<ElementType>(tMesh, tStateVector, tControl);
    TEST_ASSERT(tValue < 1e-16);
}

TEUCHOS_UNIT_TEST(StrainEnergy, UniaxialDisplacementGivesExpectedValue)
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

    std::vector<Plato::Scalar> tDispVals{0.0, 0.0, 0.0, 0.0,
                                         1.0, 0.0, 1.0, 0.0};  // uniaxial displacement field with extension of 1.0

    // Compute gold energy in Matlab:
    //      F=H+eye(3)
    //      Finv=F\eye(3)
    //      J=det(F)
    //      J23=J^(-2/3)
    //      I1Bar=J23*F(:)'*F(:)
    //      Wvol = 0.5*K*(0.5*J^2 - 0.5 - log(J))
    //      Wdev = 0.5*G*(I1Bar - 3.0)
    //      Wvol+Wdev
    constexpr Plato::Scalar tGoldValue{0.3479187954258798};  // energy is constant throughout domain with unit area

    // control of 1
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(1.0, tControl);

        const auto tValue = compute_criterion_over_mesh<ElementType>(tMesh, tDispVals, tControl);
        TEST_FLOATING_EQUALITY(tValue, tGoldValue, 1e-14);
    }

    // control of 0.2
    {
        constexpr Plato::Scalar tControlValue{0.25};
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(tControlValue, tControl);

        const auto tValue = compute_criterion_over_mesh<ElementType>(tMesh, tDispVals, tControl);
        TEST_FLOATING_EQUALITY(tValue, tControlValue * tGoldValue, 1e-14);
    }
}
}  // namespace plato::elliptic::finite_deformation_mechanics::unittest
