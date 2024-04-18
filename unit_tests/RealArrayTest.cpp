#include "util/PlatoTestHelpers.hpp"
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Sacado.hpp>

#include "ParseTools.hpp"
#include "utilities/ExpressionParser.hpp"

template <typename T>
std::pair<Plato::Scalar, Plato::Scalar> evalFunction(const Plato::Scalar aZ, T aLambda) {
  using FadType = Sacado::Fad::SFad<Plato::Scalar, 1>;

  typename Plato::Evaluator::Expression<FadType>::ArrayType tZ(/*Length=*/1);
  Kokkos::deep_copy(tZ.mData, FadType(1, 0, aZ));

  auto tResult = aLambda(tZ);
  auto tResult_Host = Kokkos::create_mirror(tResult.mData);
  Kokkos::deep_copy(tResult_Host, tResult.mData);
  return std::make_pair(tResult_Host(0).val(), tResult_Host(0).dx(0));
}

namespace PlatoUnitTests
{
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, sin) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::sin(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, sin(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, cos(Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, cos) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::cos(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, cos(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, -sin(Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, tan) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::tan(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, tan(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, pow(1.0 / cos(Z), 2.0), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, asin) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::asin(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, asin(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, 1.0 / sqrt(1 - Z * Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, acos) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::acos(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, acos(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, -1.0 / sqrt(1 - Z * Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, atan) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::atan(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, atan(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, 1.0 / (1 + Z * Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, sinh) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::sinh(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, sinh(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, cosh(Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, cosh) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::cosh(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, cosh(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, sinh(Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, tanh) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::tanh(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, tanh(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, pow(1.0 / cosh(Z), 2.0), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, asinh) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::asinh(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, asinh(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, 1.0 / sqrt(1 + Z * Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, acosh) {
  Plato::Scalar Z = 3.0;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::acosh(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, acosh(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, 1.0 / (sqrt(1 + Z) * sqrt(Z - 1)), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, atanh) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::atanh(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, atanh(Z), 1e-15);
  TEST_FLOATING_EQUALITY(deriv, 1.0 / (1.0 - Z * Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, naturalLog) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::log(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, log(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, 1.0 / Z, 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, commonLog) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::log10(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, log10(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, 1.0 / (Z * log(10)), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, exp) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::exp(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, exp(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, exp(Z), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, sqr) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return aZ * aZ; };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, Z * Z, 1e-18);
  TEST_FLOATING_EQUALITY(deriv, 2.0 * Z, 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, sqrt) {
  Plato::Scalar Z = 0.5;

  auto tFun = [](auto& aZ) { return Plato::Evaluator::Math::sqrt(aZ); };
  auto [value, deriv] = evalFunction(Z, tFun);

  TEST_FLOATING_EQUALITY(value, sqrt(Z), 1e-18);
  TEST_FLOATING_EQUALITY(deriv, 1.0 / 2.0 * pow(Z, -1.0 / 2.0), 1e-15);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_unary_negative) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/1.0);

  auto tZ = -tX;

  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);

  TEST_FLOATING_EQUALITY(tZ_Host(0), -1.0, 1e-18);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_unary_positive) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/-1.0);

  auto tZ = +tX;

  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);

  TEST_FLOATING_EQUALITY(tZ_Host(0), -1.0, 1e-18);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_operator_plus) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/1.0), tY(tLength, /*init=*/2.0);

  auto tZ = tX + tY;

  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);

  TEST_FLOATING_EQUALITY(tZ_Host(0), 3.0, 1e-18);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_operator_minus) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/3.0), tY(tLength, /*init=*/2.0);

  auto tZ = tX - tY;

  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);

  TEST_FLOATING_EQUALITY(tZ_Host(0), 1.0, 1e-18);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_operator_times) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/3.0), tY(tLength, /*init=*/2.0);

  auto tZ = tX * tY;

  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);

  TEST_FLOATING_EQUALITY(tZ_Host(0), 6.0, 1e-18);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_operator_divide) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/3.0), tY(tLength, /*init=*/2.0);

  auto tZ = tX / tY;

  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);

  TEST_FLOATING_EQUALITY(tZ_Host(0), 3.0 / 2.0, 1e-18);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_function_pow) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/3.0), tY(tLength, /*init=*/2.0);

  auto tZ = Plato::Evaluator::Math::pow(tX, tY);

  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);

  TEST_FLOATING_EQUALITY(tZ_Host(0), 9.0, 1e-18);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_function_pow2) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/3.0);

  auto tZ = Plato::Evaluator::Math::pow(tX, 2.0);

  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);

  TEST_FLOATING_EQUALITY(tZ_Host(0), 9.0, 1e-18);
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, RealArray_operator_equals) {
  int tLength = 1;
  Plato::Evaluator::RealArray<Plato::Scalar> tX(tLength, /*init=*/3.0);

  // use copy here since assignment does a shallow copy
  auto tZ = Plato::Evaluator::Math::copy(tX);

  // check assignment of tZ
  auto tZ_Host = Kokkos::create_mirror(tZ.mData);
  Kokkos::deep_copy(tZ_Host, tZ.mData);
  TEST_FLOATING_EQUALITY(tZ_Host(0), 3.0, 1e-18);

  // check assignment of tZ to a literal;
  tZ = 1.0;
  Kokkos::deep_copy(tZ_Host, tZ.mData);
  TEST_FLOATING_EQUALITY(tZ_Host(0), 1.0, 1e-18);
}
}