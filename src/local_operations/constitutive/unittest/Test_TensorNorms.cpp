#include <Teuchos_UnitTestHarness.hpp>
#include <cmath>

#include "linear_algebra/PlatoMathTypes.hpp"
#include "local_operations/constitutive/VoigtTensorL2Norm.hpp"
#include "local_operations/constitutive/VonMisesNorm.hpp"

namespace plato::local_operations::constitutive::unittest
{
TEUCHOS_UNIT_TEST(VoigtTensorL2Norm, ZeroStressGivesZeroNorm)
{
    {
        const auto tStress = Plato::Array<3>{0.0};
        TEST_ASSERT(std::fabs(Plato::VoigtTensorL2Norm{}(tStress)) < 1e-16);
    }
    {
        const auto tStress = Plato::Array<6>{0.0};
        TEST_ASSERT(std::fabs(Plato::VoigtTensorL2Norm{}(tStress)) < 1e-16);
    }
}

TEUCHOS_UNIT_TEST(VoigtTensorL2Norm, ValueMatchesExpected2D)
{
    {
        const auto tStress = Plato::Array<3>{1.0, 1.0, 1.0};
        const double tGoldValue{std::sqrt(3)};
        TEST_EQUALITY(Plato::VoigtTensorL2Norm{}(tStress), tGoldValue);
    }
    {
        const auto tStress = Plato::Array<3>{1.0, 2.0, 3.0};
        const double tGoldValue{std::sqrt(14)};
        TEST_EQUALITY(Plato::VoigtTensorL2Norm{}(tStress), tGoldValue);
    }
}

TEUCHOS_UNIT_TEST(VoigtTensorL2Norm, ValueMatchesExpected3D)
{
    {
        const auto tStress = Plato::Array<6>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        const double tGoldValue{std::sqrt(6)};
        TEST_EQUALITY(Plato::VoigtTensorL2Norm{}(tStress), tGoldValue);
    }
    {
        const auto tStress = Plato::Array<6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        const double tGoldValue{std::sqrt(91)};
        TEST_EQUALITY(Plato::VoigtTensorL2Norm{}(tStress), tGoldValue);
    }
}

TEUCHOS_UNIT_TEST(VonMisesNorm, ZeroStressGivesZeroNorm)
{
    {
        const auto tStress = Plato::Array<3>{0.0};
        TEST_ASSERT(std::fabs(Plato::VonMisesNorm{}(tStress)) < 1e-16);
    }
    {
        const auto tStress = Plato::Array<6>{0.0};
        TEST_ASSERT(std::fabs(Plato::VonMisesNorm{}(tStress)) < 1e-16);
    }
}

TEUCHOS_UNIT_TEST(VonMisesNorm, ValueMatchesExpected2D)
{
    // for plane stress: sqrt(T11*T11 - T11*T22 + T22*T22 + 3 T12*T12
    {
        const auto tStress = Plato::Array<3>{1.0, 1.0, 1.0};
        constexpr double tGoldValue{2.0};
        TEST_EQUALITY(Plato::VonMisesNorm{}(tStress), tGoldValue);
    }
    {
        const auto tStress = Plato::Array<3>{1.0, 2.0, 3.0};
        const double tGoldValue{std::sqrt(30)};
        TEST_EQUALITY(Plato::VonMisesNorm{}(tStress), tGoldValue);
    }
}

TEUCHOS_UNIT_TEST(VonMisesNorm, ValueMatchesExpected3D)
{
    // for 3D: sqrt(0.5*[(T11-T22)^2 + (T22-T33)^2 + (T33-T11)^2)] + 3*(T12*T12 + T23*T23 + T31*T31))
    {
        const auto tStress = Plato::Array<6>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        constexpr double tGoldValue{3.0};
        TEST_EQUALITY(Plato::VonMisesNorm{}(tStress), tGoldValue);
    }
    {
        const auto tStress = Plato::Array<6>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        const double tGoldValue{std::sqrt(234)};
        TEST_EQUALITY(Plato::VonMisesNorm{}(tStress), tGoldValue);
    }
}
}  // namespace plato::local_operations::constitutive::unittest
