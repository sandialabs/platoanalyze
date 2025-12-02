#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "core_types/PlatoTypes.hpp"
#include "domain/SpatialModel.hpp"
#include "element/MechanicsElement.hpp"
#include "element/Tet4.hpp"
#include "element/Tri3.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "local_operations/constitutive/TensorNormFactory.hpp"
#include "local_operations/constitutive/VonMisesYieldFunction.hpp"
#include "mesh/PlatoMesh.hpp"
#include "problem/Mechanics.hpp"
#include "problem/elliptic/PhysicsScalarFunction.hpp"
#include "test_utilities/PlatoMeshTestHelpers.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::unittest
{
namespace
{
constexpr Plato::OrdinalType kDefaultVoigtLength{6};
const std::string kPNormCriterionName{"My P-Norm"};

template <typename ElementType>
struct CreatePNormCriterion
{
    auto operator()(const Plato::SpatialModel& aSpatialModel,
                    Plato::DataMap& aDataMap,
                    Teuchos::ParameterList& aParameterList) const
    {
        return Plato::Elliptic::PhysicsScalarFunction<Plato::Mechanics<typename ElementType::TopoElementType>>(
            aSpatialModel, aDataMap, aParameterList, kPNormCriterionName);
    }
};

struct TestEvaluationTypes
{
    using StateScalarType = Plato::Scalar;
    using ControlScalarType = Plato::Scalar;
    using ConfigScalarType = Plato::Scalar;
    using ResultScalarType = Plato::Scalar;
};

[[nodiscard]] Teuchos::ParameterList create_parameter_list_with_p_norm(const std::string& aNormType)
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");

    tParameterList.sublist("Criteria").sublist(kPNormCriterionName).set("Type", "Scalar Function");
    tParameterList.sublist("Criteria").sublist(kPNormCriterionName).set("Scalar Function Type", "Stress P-Norm");
    tParameterList.sublist("Criteria").sublist(kPNormCriterionName).set("Exponent", 1.0);
    tParameterList.sublist("Criteria").sublist(kPNormCriterionName).sublist("Normalize").set("Type", aNormType);
    tParameterList.sublist("Criteria").sublist(kPNormCriterionName).sublist("Normalize").set("Volume Scaling", false);

    tParameterList.sublist("Criteria").sublist(kPNormCriterionName).sublist("Penalty Function").set("Type", "SIMP");
    tParameterList.sublist("Criteria").sublist(kPNormCriterionName).sublist("Penalty Function").set("Exponent", 0.5);
    tParameterList.sublist("Criteria")
        .sublist(kPNormCriterionName)
        .sublist("Penalty Function")
        .set("Minimum Value", 0.0);
    return tParameterList;
}

void append_domain_and_material(Teuchos::ParameterList& aParameterList,
                                const std::string& aBlockName,
                                const std::string& aMaterialName,
                                const double aYoungsModulus,
                                const double aPoissonRatio)
{
    aParameterList.sublist("Spatial Model").sublist("Domains").sublist(aBlockName).set("Element Block", aBlockName);
    aParameterList.sublist("Spatial Model").sublist("Domains").sublist(aBlockName).set("Material Model", aMaterialName);

    aParameterList.sublist("Material Models")
        .sublist(aMaterialName)
        .sublist("Isotropic Linear Elastic")
        .set("Youngs Modulus", aYoungsModulus);
    aParameterList.sublist("Material Models")
        .sublist(aMaterialName)
        .sublist("Isotropic Linear Elastic")
        .set("Poissons Ratio", aPoissonRatio);
}
}  // namespace

TEUCHOS_UNIT_TEST(TensorNormFactory, ThrowIfNormalizeNotIncluded)
{
    Plato::TensorNormFactory<kDefaultVoigtLength, TestEvaluationTypes> tNormFactory;

    Teuchos::ParameterList tPNormCriterionParameters;
    tPNormCriterionParameters.setName("My P-Norm");
    tPNormCriterionParameters.set("Exponent", 1.0);
    TEST_THROW(tNormFactory.create(tPNormCriterionParameters), std::runtime_error);
}

TEUCHOS_UNIT_TEST(TensorNormFactory, ThrowIfInvalidNormTypeSpecified)
{
    constexpr Plato::OrdinalType tVoigtLength{6};
    Plato::TensorNormFactory<kDefaultVoigtLength, TestEvaluationTypes> tNormFactory;

    const std::string tInvalidNorm{"Spaghetti-Os"};

    Teuchos::ParameterList tPNormCriterionParameters;
    tPNormCriterionParameters.setName("My P-Norm");
    tPNormCriterionParameters.set("Exponent", 1.0);
    tPNormCriterionParameters.sublist("Normalize").set("Type", tInvalidNorm);
    TEST_THROW(tNormFactory.create(tPNormCriterionParameters), std::runtime_error);
}

TEUCHOS_UNIT_TEST(VonMisesPNorm, ZeroDisplacementGivesZeroValue)
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

    auto tParameterList = create_parameter_list_with_p_norm("Von Mises");
    append_domain_and_material(tParameterList,
                               /*aBlockName=*/"body",
                               /*aMaterialName=*/"chapstick",
                               /*aYoungsModulus=*/1.0, /*aPoissonRatio=*/0.3);

    const auto tValue = Plato::TestHelpers::compute_criterion_over_mesh<ElementType>(
        CreatePNormCriterion<ElementType>{}, tMesh, tParameterList,
        Plato::TestHelpers::single_step_solutions(tStateVector), tControl);
    TEST_ASSERT(std::fabs(tValue) < 1e-16);
}

TEUCHOS_UNIT_TEST(VonMisesPNorm, UniaxialDisplacementGivesExpectedValue)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 2;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();
    const auto tNumElements = tMesh->NumElements();

    constexpr double tExtension{1.0};
    const Plato::Matrix<tSpatialDims, tSpatialDims, Plato::Scalar> tAppliedDisplacementGradient{
        tExtension, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto tDisplacementField =
        Plato::TestHelpers::create_linear_displacement_field(tMesh, tAppliedDisplacementGradient);

    constexpr double tYoungsModulus{86.0};
    constexpr double tPoissonRatio{0.37};
    constexpr double tConstant = tYoungsModulus / (1 + tPoissonRatio) / (1 - 2 * tPoissonRatio);
    const auto tGoldStress = Plato::Array<6>{tConstant * (1 - tPoissonRatio) * tExtension,
                                             tConstant * tPoissonRatio * tExtension,
                                             tConstant * tPoissonRatio * tExtension,
                                             0.0,
                                             0.0,
                                             0.0};

    const Plato::VonMisesYieldFunction<3, 6> tComputeVonMises;
    Plato::Scalar tGoldVonMisesStress{};
    tComputeVonMises(tGoldStress, tGoldVonMisesStress);
    tGoldVonMisesStress *= tNumElements;  // p-norm is summed over elements

    auto tParameterList = create_parameter_list_with_p_norm("Von Mises");
    append_domain_and_material(tParameterList,
                               /*aBlockName=*/"body",
                               /*aMaterialName=*/"chapstick",
                               /*aYoungsModulus=*/tYoungsModulus, /*aPoissonRatio=*/tPoissonRatio);

    // control of 1
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(1.0, tControl);

        const auto tValue = Plato::TestHelpers::compute_criterion_over_mesh<ElementType>(
            CreatePNormCriterion<ElementType>{}, tMesh, tParameterList,
            Plato::TestHelpers::single_step_solutions(tDisplacementField), tControl);
        TEST_FLOATING_EQUALITY(tValue, tGoldVonMisesStress, 1e-14);
    }
    // control of 0.86
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        constexpr double tControlValue{0.86};
        Plato::blas1::fill(tControlValue, tControl);

        const auto tValue = Plato::TestHelpers::compute_criterion_over_mesh<ElementType>(
            CreatePNormCriterion<ElementType>{}, tMesh, tParameterList,
            Plato::TestHelpers::single_step_solutions(tDisplacementField), tControl);
        tGoldVonMisesStress *= std::sqrt(tControlValue);  // SIMP exponent of 0.5 is specified in input
        TEST_FLOATING_EQUALITY(tValue, tGoldVonMisesStress, 1e-14);
    }
}

TEUCHOS_UNIT_TEST(VonMisesPNorm, UniaxialDisplacementGivesExpectedValueTwoBlockTri)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    const auto tMesh = Plato::TestHelpers::TwoBlockTriMeshRAII{};
    const auto tNumNodes = tMesh.mMesh->NumNodes();
    const auto tNumElements = tMesh.mMesh->NumElements();

    constexpr double tExtension{1.0};
    const Plato::Matrix<tSpatialDims, tSpatialDims, Plato::Scalar> tAppliedDisplacementGradient{tExtension, 0.0, 0.0,
                                                                                                0.0};
    const auto tDisplacementField =
        Plato::TestHelpers::create_linear_displacement_field(tMesh.mMesh, tAppliedDisplacementGradient);

    constexpr double tYoungsModulus{86.0};
    constexpr double tPoissonRatio{0.37};
    constexpr double tConstant = tYoungsModulus / (1 + tPoissonRatio) / (1 - 2 * tPoissonRatio);
    const auto tGoldStress =
        Plato::Array<3>{tConstant * (1 - tPoissonRatio) * tExtension, tConstant * tPoissonRatio * tExtension, 0.0};

    const Plato::VonMisesYieldFunction<2, 3> tComputeVonMises;
    Plato::Scalar tGoldVonMisesStress{};
    tComputeVonMises(tGoldStress, tGoldVonMisesStress);
    tGoldVonMisesStress *= 3.0 * tNumElements /
                           7.0;  // p-norm is summed over elements, 3 of the 7 elements in this test mesh are in block 1

    auto tParameterList = create_parameter_list_with_p_norm("Von Mises");
    tParameterList.sublist("Criteria")
        .sublist("My P-Norm")
        .set<Teuchos::Array<std::string>>("Domains", Teuchos::Array<std::string>{"BLOCK_1"});
    append_domain_and_material(tParameterList,
                               /*aBlockName=*/"BLOCK_1",
                               /*aMaterialName=*/"chapstick",
                               /*aYoungsModulus=*/tYoungsModulus, /*aPoissonRatio=*/tPoissonRatio);
    append_domain_and_material(tParameterList,
                               /*aBlockName=*/"BLOCK_2",
                               /*aMaterialName=*/"vaseline",
                               /*aYoungsModulus=*/10.0 * tYoungsModulus, /*aPoissonRatio=*/0.5 * tPoissonRatio);

    // control of 1
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(1.0, tControl);

        const auto tValue = Plato::TestHelpers::compute_criterion_over_mesh<ElementType>(
            CreatePNormCriterion<ElementType>{}, tMesh.mMesh, tParameterList,
            Plato::TestHelpers::single_step_solutions(tDisplacementField), tControl);
        TEST_FLOATING_EQUALITY(tValue, tGoldVonMisesStress, 1e-14);
    }
}

TEUCHOS_UNIT_TEST(VoigtTensorL2PNorm, UniaxialDisplacementGivesExpectedValue)
{
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;
    constexpr Plato::OrdinalType tSpatialDims = ElementType::mNumSpatialDims;

    constexpr Plato::OrdinalType tMeshWidth = 2;
    const auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    const auto tNumNodes = tMesh->NumNodes();
    const auto tNumElements = tMesh->NumElements();

    constexpr double tExtension{1.0};
    const Plato::Matrix<tSpatialDims, tSpatialDims, Plato::Scalar> tAppliedDisplacementGradient{
        tExtension, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const auto tDisplacementField =
        Plato::TestHelpers::create_linear_displacement_field(tMesh, tAppliedDisplacementGradient);

    constexpr double tYoungsModulus{86.0};
    constexpr double tPoissonRatio{0.37};
    constexpr double tConstant = tYoungsModulus / (1 + tPoissonRatio) / (1 - 2 * tPoissonRatio);
    const auto tGoldStress = Plato::Array<6>{tConstant * (1 - tPoissonRatio) * tExtension,
                                             tConstant * tPoissonRatio * tExtension,
                                             tConstant * tPoissonRatio * tExtension,
                                             0.0,
                                             0.0,
                                             0.0};

    auto tGoldVoigtTensorNorm = tNumElements * Plato::norm(tGoldStress);  // p-norm is summed over elements

    auto tParameterList = create_parameter_list_with_p_norm("Voigt Tensor L2");
    append_domain_and_material(tParameterList,
                               /*aBlockName=*/"body",
                               /*aMaterialName=*/"chapstick",
                               /*aYoungsModulus=*/tYoungsModulus, /*aPoissonRatio=*/tPoissonRatio);

    // control of 1
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        Plato::blas1::fill(1.0, tControl);

        const auto tValue = Plato::TestHelpers::compute_criterion_over_mesh<ElementType>(
            CreatePNormCriterion<ElementType>{}, tMesh, tParameterList,
            Plato::TestHelpers::single_step_solutions(tDisplacementField), tControl);
        TEST_FLOATING_EQUALITY(tValue, tGoldVoigtTensorNorm, 1e-14);
    }
    // control of 0.88
    {
        const Plato::ScalarVector tControl("control", tNumNodes);
        constexpr double tControlValue{0.88};
        Plato::blas1::fill(tControlValue, tControl);

        const auto tValue = Plato::TestHelpers::compute_criterion_over_mesh<ElementType>(
            CreatePNormCriterion<ElementType>{}, tMesh, tParameterList,
            Plato::TestHelpers::single_step_solutions(tDisplacementField), tControl);
        tGoldVoigtTensorNorm *= std::sqrt(tControlValue);  // SIMP exponent of 0.5 is specified in input
        TEST_FLOATING_EQUALITY(tValue, tGoldVoigtTensorNorm, 1e-14);
    }
}
}  // namespace plato::unittest
