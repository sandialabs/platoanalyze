#include <Teuchos_UnitTestHarness.hpp>
#include <vector>

#include "BLAS1.hpp"
#include "MechanicsElement.hpp"
#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoTestHelpers.hpp"
#include "PlatoTypes.hpp"
#include "Solutions.hpp"
#include "SpatialModel.hpp"
#include "Tri3.hpp"
#include "elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"
#include "elliptic/finite_deformation_mechanics/VarianceFunction.hpp"

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

    tParameterList.sublist("Criteria").sublist("Strain Variance").set("Type", "Variance Function");
    tParameterList.sublist("Criteria").sublist("Strain Variance").set("Field Variable", "Strain Invariant");
    tParameterList.sublist("Criteria").sublist("Strain Variance").sublist("Penalty Function").set("Type", "SIMP");
    tParameterList.sublist("Criteria").sublist("Strain Variance").sublist("Penalty Function").set("Exponent", 1.0);
    tParameterList.sublist("Criteria")
        .sublist("Strain Variance")
        .sublist("Penalty Function")
        .set("Minimum Value", 1e-16);

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
auto get_variance_criterion_for_mesh(const Plato::Mesh& aMesh)
    -> VarianceFunction<FiniteDeformationMechanics<typename ElementType::TopoElementType>>
{
    Teuchos::ParameterList tParamList = create_param_list();
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(aMesh, tParamList, tDataMap);
    const std::string tCriterionName = "Strain Variance";

    return VarianceFunction<FiniteDeformationMechanics<typename ElementType::TopoElementType>>{
        tSpatialModel, tDataMap, tParamList, tCriterionName};
}

constexpr Plato::OrdinalType kGradientSize{4};
template <typename ElementType>
void test_value_and_gradient_against_gold(const Plato::Mesh& aMesh,
                                          const Plato::Solutions& aSolution,
                                          const Plato::ScalarVector& aControl,
                                          const Plato::Scalar aGoldValue,
                                          const std::array<Plato::Scalar, kGradientSize>& aGoldGradient,
                                          Teuchos::FancyOStream& aOutStream,
                                          bool& aSuccess)
{
    const auto tVarianceCriterion = get_variance_criterion_for_mesh<ElementType>(aMesh);

    const auto tValue = tVarianceCriterion.value(aSolution, aControl);
    TEUCHOS_TEST_FLOATING_EQUALITY(tValue, aGoldValue, 1e-13, aOutStream, aSuccess);

    const auto tGradient = tVarianceCriterion.gradient_z(aSolution, aControl);
    const auto tGradientHost = Plato::TestHelpers::get(tGradient);
    TEUCHOS_TEST_ASSERT(tGradientHost.size() == kGradientSize, aOutStream, aSuccess);
    for (Plato::OrdinalType tIndex = 0; tIndex < tGradientHost.size(); tIndex++)
    {
        TEUCHOS_TEST_FLOATING_EQUALITY(tGradientHost(tIndex), aGoldGradient[tIndex], 1e-12, aOutStream, aSuccess);
    }
}
}  // namespace

TEUCHOS_UNIT_TEST(Variance, UniaxialDisplacementGivesZeroVariance)
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

    std::vector<Plato::Scalar> tDispVals{0.0, 0.01, 0.0, -0.01,
                                         1.0, 0.01, 1.0, -0.01};  // uniaxial displacement field with extension of 1.0
    const auto tSolution = Plato::TestHelpers::single_step_solutions_from_vector(tDispVals);

    constexpr Plato::Scalar tGoldVariance{0.0};
    // control of 1
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(1.0, tControl);
        const auto tVarianceCriterion = get_variance_criterion_for_mesh<ElementType>(tMesh);
        const auto tVariance = tVarianceCriterion.value(tSolution, tControl);
        TEST_FLOATING_EQUALITY(tVariance, tGoldVariance, 1e-14);
    }
}

TEUCHOS_UNIT_TEST(Variance, NonHomogeneousDisplacementAndGradientZMatchAnalytic)
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

    // MATLAB code for gold value:
    // clang-format off
    // p = 0.25;
    // disp_field = [0.0, 0.02, 0.0, -0.02, 0.0, -0.02, 0.0, 0.02];
    // nele = 2;
    // shapeVal = 1/3; % shape function values are the same for each node
    // shapeGrads=[-1,-1;1,0;0,1];

    // ele1coords=[0.0, 0.0; 1.0, 0.0; 1.0, 1.0];
    // ele2coords=[0.0, 0.0; 1.0, 1.0; 0.0, 1.0];
    // ele1Jac=[shapeGrads(:,1)'*ele1coords(:,1), shapeGrads(:,2)'*ele1coords(:,1); shapeGrads(:,1)'*ele1coords(:,2), shapeGrads(:,2)'*ele1coords(:,2)]\eye(2);
    // ele2Jac=[shapeGrads(:,1)'*ele2coords(:,1),shapeGrads(:,2)'*ele2coords(:,1); shapeGrads(:,1)'*ele2coords(:,2), shapeGrads(:,2)'*ele2coords(:,2)]\eye(2);
    // ele1shapeGradsX=shapeGrads*ele1Jac;
    // ele2shapeGradsX=shapeGrads*ele2Jac;

    // ele1disp=[disp_field(1:2); disp_field(5:6); disp_field(7:8)];
    // ele2disp=[disp_field(1:2); disp_field(7:8); disp_field(3:4)];
    // ele1F=[ele1shapeGradsX(:,1)'*ele1disp(:,1), ele1shapeGradsX(:,2)'*ele1disp(:,1), 0; ele1shapeGradsX(:,1)'*ele1disp(:,2), ele1shapeGradsX(:,2)'*ele1disp(:,2), 0; 0, 0, 0] + eye(3);
    // ele2F=[ele2shapeGradsX(:,1)'*ele2disp(:,1), ele2shapeGradsX(:,2)'*ele2disp(:,1), 0; ele2shapeGradsX(:,1)'*ele2disp(:,2), ele2shapeGradsX(:,2)'*ele2disp(:,2), 0; 0, 0, 0] + eye(3);
    // ele1I1=p*ele1F(:)'*ele1F(:);
    // ele2I1=p*ele2F(:)'*ele2F(:);
    // I1mean = (ele1I1+ele2I1)/nele;
    // I1var=((ele1I1-I1mean)^2+(ele2I1-I1mean)^2)/nele

    // dele1I1_dz = shapeVal*ele1I1/p;
    // dele2I1_dz = shapeVal*ele2I1/p;
    // dI1var_dz_ele1 = (2 * (ele1I1 - I1mean) / nele) * dele1I1_dz;
    // dI1var_dz_ele2 = (2 * (ele2I1 - I1mean) / nele) * dele2I1_dz;
    // gradient = [dI1var_dz_ele1+dI1var_dz_ele2, dI1var_dz_ele2, dI1var_dz_ele1, dI1var_dz_ele1+dI1var_dz_ele2]
    // clang-format on

    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();

    std::vector<Plato::Scalar> tDispVals{0.0, 0.02, 0.0, -0.02, 0.0, -0.02, 0.0, 0.02};
    const auto tSolution = Plato::TestHelpers::single_step_solutions_from_vector(tDispVals);

    // control of 1
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(1.0, tControl);
        constexpr Plato::Scalar tGoldVariance{6.400000000000012e-3};
        constexpr std::array<Plato::Scalar, kGradientSize> tGoldGradient{4.266666666666682e-3, -7.795200000000006e-2,
                                                                         8.221866666666675e-2, 4.266666666666682e-3};
        test_value_and_gradient_against_gold<ElementType>(tMesh, tSolution, tControl, tGoldVariance, tGoldGradient, out,
                                                          success);
    }

    // control of 0.25
    {
        constexpr Plato::Scalar tControlValue{0.25};
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(tControlValue, tControl);
        constexpr Plato::Scalar tGoldVariance{4.000000000000007e-4};
        constexpr std::array<Plato::Scalar, kGradientSize> tGoldGradient{1.066666666666671e-3, -1.948800000000002e-2,
                                                                         2.055466666666669e-2, 1.066666666666671e-3};
        test_value_and_gradient_against_gold<ElementType>(tMesh, tSolution, tControl, tGoldVariance, tGoldGradient, out,
                                                          success);
    }
}
}  // namespace plato::elliptic::finite_deformation_mechanics::unittest
