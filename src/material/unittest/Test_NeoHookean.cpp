#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <cmath>

#include "core_types/PlatoTypes.hpp"
#include "linear_algebra/PlatoMathTypes.hpp"
#include "material/NeoHookeanModel.hpp"

namespace plato::composable_function_objects::material::unittest
{
namespace
{
constexpr Plato::OrdinalType kTensorDimension{3};
constexpr NeoHookeanParameters kMaterialParameters{/*mBulkModulus=*/0.5, /*mShearModulus=*/0.375};

auto cauchy_stress_from_first_piola_kirchhoff_stress(const Plato::Matrix<kTensorDimension, kTensorDimension>& aP,
                                                     const Plato::Matrix<kTensorDimension, kTensorDimension>& aF)
    -> Plato::Matrix<kTensorDimension, kTensorDimension>
{
    const auto tFT = Plato::transpose(aF);
    const auto tJ = Plato::determinant(aF);
    return Plato::times(1.0 / tJ, Plato::times(aP, tFT));
}

void check_stress_with_gold(const Plato::Matrix<kTensorDimension, kTensorDimension>& aStress,
                            const Plato::Matrix<kTensorDimension, kTensorDimension>& aGoldStress,
                            Teuchos::FancyOStream& aOutStream,
                            bool& aSuccess)
{
    for (Plato::OrdinalType i = 0; i < kTensorDimension; i++)
    {
        for (Plato::OrdinalType j = 0; j < kTensorDimension; j++)
        {
            TEUCHOS_TEST_FLOATING_EQUALITY(aStress(i, j), aGoldStress(i, j), 1e-14, aOutStream, aSuccess);
        }
    }
}

}  // namespace

TEUCHOS_UNIT_TEST(NeoHookean, EnergyAndStressAreZeroForIdentityDeformationGradient)
{
    const auto tDeformationGradient = Plato::identity<kTensorDimension>();

    NeoHookeanModel tMaterialModel{kMaterialParameters};

    Plato::Scalar tEnergy{0.0};
    tMaterialModel.energy(tDeformationGradient, tEnergy);
    TEST_ASSERT(tEnergy < 1e-14);

    Plato::Matrix<kTensorDimension, kTensorDimension> tStress(0.0);
    tMaterialModel.stress(tDeformationGradient, tStress);

    for (Plato::OrdinalType i = 0; i < kTensorDimension; i++)
    {
        for (Plato::OrdinalType j = 0; j < kTensorDimension; j++)
        {
            TEST_ASSERT(tStress(i, j) < 1e-14);
        }
    }
}

TEUCHOS_UNIT_TEST(NeoHookean, UniaxialAnalyticSolution)
{
    // uniaxial deformation gradient
    constexpr Plato::Scalar tExtension{1.0};
    constexpr Plato::Scalar tDeformationGradient11 = tExtension + 1.0;
    const Plato::Matrix<kTensorDimension, kTensorDimension> tUniaxialDeformationGradient{
        tDeformationGradient11, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

    NeoHookeanModel tMaterialModel{kMaterialParameters};

    Plato::Scalar tEnergy{0.0};
    tMaterialModel.energy(tUniaxialDeformationGradient, tEnergy);
    // Compute gold energy in Matlab:
    //      F=H+eye(3)
    //      Finv=F\eye(3)
    //      J=det(F)
    //      J23=J^(-2/3)
    //      I1Bar=J23*F(:)'*F(:)
    //      Wvol = 0.5*K*(0.5*J^2 - 0.5 - log(J))
    //      Wdev = 0.5*G*(I1Bar - 3.0)
    //      Wvol+Wdev
    constexpr Plato::Scalar tGoldEnergy{0.3479187954258798};
    TEST_FLOATING_EQUALITY(tEnergy, tGoldEnergy, 1e-14);

    Plato::Matrix<kTensorDimension, kTensorDimension> tStress(0.0);
    tMaterialModel.stress(tUniaxialDeformationGradient, tStress);
    const auto tCauchyStress = cauchy_stress_from_first_piola_kirchhoff_stress(tStress, tUniaxialDeformationGradient);

    const auto tGoldCauchyStress11 =
        0.5 * kMaterialParameters.mBulkModulus * (tDeformationGradient11 - 1.0 / tDeformationGradient11) +
        2.0 / 3.0 * kMaterialParameters.mShearModulus * (tDeformationGradient11 * tDeformationGradient11 - 1.0) *
            std::pow(tDeformationGradient11, -5.0 / 3.0);
    const auto tGoldCauchyStress22 =
        0.5 * kMaterialParameters.mBulkModulus * (tDeformationGradient11 - 1.0 / tDeformationGradient11) -
        1.0 / 3.0 * kMaterialParameters.mShearModulus * (tDeformationGradient11 * tDeformationGradient11 - 1.0) *
            std::pow(tDeformationGradient11, -5.0 / 3.0);
    const auto tGoldCauchyStress33 = tGoldCauchyStress22;
    const Plato::Matrix<kTensorDimension, kTensorDimension> tGoldCauchyStress{
        tGoldCauchyStress11, 0.0, 0.0, 0.0, tGoldCauchyStress22, 0.0, 0.0, 0.0, tGoldCauchyStress33};

    check_stress_with_gold(tCauchyStress, tGoldCauchyStress, out, success);
}

TEUCHOS_UNIT_TEST(NeoHookean, SimpleShearAnalyticSolution)
{
    // simple shear deformation gradient
    constexpr Plato::Scalar tShear{1.0};
    const Plato::Matrix<kTensorDimension, kTensorDimension> tUniaxialDeformationGradient{1.0, tShear, 0.0, 0.0, 1.0,
                                                                                         0.0, 0.0,    0.0, 1.0};

    NeoHookeanModel tMaterialModel{kMaterialParameters};

    Plato::Scalar tEnergy{0.0};
    tMaterialModel.energy(tUniaxialDeformationGradient, tEnergy);
    // Compute gold energy in Matlab:
    //      F=H+eye(3)
    //      Finv=F\eye(3)
    //      J=det(F)
    //      J23=J^(-2/3)
    //      I1Bar=J23*F(:)'*F(:)
    //      Wvol = 0.5*K*(0.5*J^2 - 0.5 - log(J))
    //      Wdev = 0.5*G*(I1Bar - 3.0)
    //      Wvol+Wdev
    constexpr Plato::Scalar tGoldEnergy{0.1875};
    TEST_FLOATING_EQUALITY(tEnergy, tGoldEnergy, 1e-14);

    Plato::Matrix<kTensorDimension, kTensorDimension> tStress(0.0);
    tMaterialModel.stress(tUniaxialDeformationGradient, tStress);

    const auto tCauchyStress = cauchy_stress_from_first_piola_kirchhoff_stress(tStress, tUniaxialDeformationGradient);

    const auto tGoldCauchyStress11 = 2.0 / 3.0 * kMaterialParameters.mShearModulus * tShear * tShear;
    const auto tGoldCauchyStress22 = -1.0 / 3.0 * kMaterialParameters.mShearModulus * tShear * tShear;
    const auto tGoldCauchyStress33 = tGoldCauchyStress22;
    const auto tGoldCauchyStress12 = kMaterialParameters.mShearModulus * tShear;
    const Plato::Matrix<kTensorDimension, kTensorDimension> tGoldCauchyStress{
        tGoldCauchyStress11, tGoldCauchyStress12, 0.0, tGoldCauchyStress12, tGoldCauchyStress22, 0.0, 0.0, 0.0,
        tGoldCauchyStress33};

    check_stress_with_gold(tCauchyStress, tGoldCauchyStress, out, success);
}
}  // namespace plato::composable_function_objects::material::unittest
