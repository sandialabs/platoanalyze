#include <Sacado.hpp>
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "parsing/ExpressionParser.hpp"
#include "parsing/ParseTools.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

template <typename RealType>
Plato::Evaluator::Expression<RealType> getTestExpression()
{
    constexpr int cLength = 1;

    Plato::Evaluator::Expression<RealType> ob(cLength);

    typename Plato::Evaluator::Expression<RealType>::ArrayType tX(cLength, 1.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tY(cLength, 2.0);

    ob.set("xVar1", tX);
    ob.set("xVar2", tY);

    return ob;
}

template <typename RealType, typename FunctorT>
bool testExpression(std::string aExpression, FunctorT aFunctor, Plato::Scalar aTolerance = 1e-10)
{
    constexpr Plato::Scalar cZ = 0.5;

    constexpr Plato::OrdinalType cLength = 1;

    Plato::Evaluator::Expression<RealType> ob(cLength);

    typename Plato::Evaluator::Expression<RealType>::ArrayType tZ(cLength, cZ);
    ob.set("Z", tZ);

    auto tResult = ob.evaluate(aExpression);

    auto tResult_Host = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tResult_Host, tResult);

    Plato::Scalar result = aFunctor(cZ);

    std::cout << std::endl;

    for (int i = 0; i < cLength; ++i)
    {
        std::cout.precision(25);
        std::cout << "exp=" << tResult_Host(i) << " eq=" << result << "  diff=" << tResult_Host(i) - result
                  << std::endl;
    }

    bool tPassed = true;
    for (int i = 0; i < cLength; ++i)
    {
        tPassed = tPassed && (fabs(tResult_Host(i) - result) < aTolerance);
    }
    return tPassed;
}
template <typename RealType>
bool testDerivative(std::string aExpression, RealType aDeriv, Plato::Scalar aTolerance = 1e-10)
{
    constexpr Plato::Scalar cZ = 0.5;

    constexpr Plato::OrdinalType cLength = 1;

    using FadType = Sacado::Fad::SFad<RealType, 1>;

    Plato::Evaluator::Expression<FadType> ob(cLength);

    Plato::ScalarVectorT<FadType> tZ("Z", cLength);
    Kokkos::deep_copy(tZ, FadType(1, 0, cZ));

    ob.set("Z", tZ);

    auto tResult = ob.evaluate(aExpression);

    auto tResult_Host = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tResult_Host, tResult);

    std::cout << std::endl;

    for (int i = 0; i < cLength; ++i)
    {
        std::cout.precision(25);
        std::cout << "computed=" << tResult_Host(i).dx(0) << " gold=" << aDeriv
                  << "  diff=" << tResult_Host(i).dx(0) - aDeriv << std::endl;
    }

    bool tPassed = true;
    for (int i = 0; i < cLength; ++i)
    {
        tPassed = tPassed && (fabs(tResult_Host(i).dx(0) - aDeriv) / fabs(aDeriv) < aTolerance);
    }
    return tPassed;
}

namespace PlatoUnitTests
{
/******************************************************************************/
/*!
  \brief Unit tests for Plato::Evaluator::Expression
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_double)
{
    using RealType = double;

    constexpr int cLength = 1;

    Plato::Evaluator::Expression<RealType> ob(cLength);

    typename Plato::Evaluator::Expression<RealType>::ArrayType tX(cLength, 1.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tY(cLength, 2.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tZ(cLength, 3.0);

    ob.set("xVar", tX);
    ob.set("yVar", tY);
    ob.set("zVar", tZ);

    auto tAnswer = ob.evaluate("(xVar+yVar)^zVar");

    auto tAnswer_Host = Kokkos::create_mirror_view(tAnswer);
    Kokkos::deep_copy(tAnswer_Host, tAnswer);

    TEST_FLOATING_EQUALITY(tAnswer_Host[0], 27, 1e-18);
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::Evaluator::Expression
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_2Vars)
{
    using RealType = double;

    constexpr int cLength = 1;

    Plato::Evaluator::Expression<RealType> ob(cLength);

    typename Plato::Evaluator::Expression<RealType>::ArrayType tX(cLength, 1.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tY(cLength, 2.0);

    ob.set("xVar1", tX);
    ob.set("xVar2", tY);

    auto tAnswer = ob.evaluate("xVar1+xVar2");

    auto tAnswer_Host = Kokkos::create_mirror_view(tAnswer);
    Kokkos::deep_copy(tAnswer_Host, tAnswer);

    TEST_FLOATING_EQUALITY(tAnswer_Host[0], 3, 1e-18);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception when an undefined variable
  is referenced.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_VariableNotDefined)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar1+xVar2+notDefined"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception when increment is attempted
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_Increment)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar1++"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception when decrement is attempted
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_Decrement)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar1--"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception when double caret encountered
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_DoubleCaret)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar1^^2"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception when double star encountered
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_DoubleStar)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar1**2"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception on plus minus
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_PlusMinus)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar1+-"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception when an empty expression
  is evaluated.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_ExpressionIsEmpty)
{
    TEST_THROW(getTestExpression<double>().evaluate(""), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception when parantheses are
  not balanced.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_BadParentheses)
{
    TEST_THROW(getTestExpression<double>().evaluate("(xVar1+xVar2"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception when a non-existent
  is called.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_NotAFunction)
{
    TEST_THROW(getTestExpression<double>().evaluate("NOT(xVar1+xVar2)"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception for a bad assignment.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_SyntaxError_BadAssignment)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar1+xVar2 = 1)"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception for superfluous space.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_SyntaxError_ExtraSpace)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar = 1 1)"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test that the Evaluator throws an exception for a missing operator
  is called.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_Exception_MissingOperator)
{
    TEST_THROW(getTestExpression<double>().evaluate("xVar(1+1)"), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief Test with unnecessary parenthesis
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_ExtraParen)
{
    Plato::Evaluator::Expression<double> ob(/*Length=*/1);

    auto tAnswer = ob.evaluate("((1+(1)))");

    auto tAnswer_Host = Kokkos::create_mirror_view(tAnswer);
    Kokkos::deep_copy(tAnswer_Host, tAnswer);

    TEST_FLOATING_EQUALITY(tAnswer_Host[0], 2, 1e-18);
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::Evaluator::Expression

  Test support for intermediate variables, i.e., wVar below.

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_double_multi)
{
    using RealType = double;

    constexpr int cLength = 1;

    Plato::Evaluator::Expression<RealType> ob(cLength);

    typename Plato::Evaluator::Expression<RealType>::ArrayType tX(cLength, 1.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tY(cLength, 2.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tZ(cLength, 3.0);

    ob.set("xVar", tX);
    ob.set("yVar", tY);
    ob.set("zVar", tZ);

    ob.evaluate("wVar=xVar+yVar");
    auto tAnswer = ob.evaluate("wVar^zVar");

    auto tAnswer_Host = Kokkos::create_mirror_view(tAnswer);
    Kokkos::deep_copy(tAnswer_Host, tAnswer);

    TEST_FLOATING_EQUALITY(tAnswer_Host[0], 27, 1e-18);
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::Evaluator::Expression

  Test support for constant literals in the expressions.

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_double_literals)
{
    using RealType = double;

    constexpr int cLength = 1;

    Plato::Evaluator::Expression<RealType> ob(cLength);

    typename Plato::Evaluator::Expression<RealType>::ArrayType tY(cLength, 2.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tZ(cLength, 3.0);

    ob.set("yVar", tY);
    ob.set("zVar", tZ);

    ob.evaluate("wVar=1.0+yVar");
    auto tAnswer = ob.evaluate("wVar^zVar");

    auto wVar = ob.get("wVar");

    auto wVar_Host = Kokkos::create_mirror_view(wVar);
    Kokkos::deep_copy(wVar_Host, wVar);

    TEST_FLOATING_EQUALITY(wVar_Host[0], 3.0, 1e-18);

    auto tAnswer_Host = Kokkos::create_mirror_view(tAnswer);
    Kokkos::deep_copy(tAnswer_Host, tAnswer);

    TEST_FLOATING_EQUALITY(tAnswer_Host[0], 27, 1e-18);
}
/******************************************************************************/
/*!
  \brief Unit tests for Plato::Evaluator::Expression
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_float)
{
    using RealType = float;

    constexpr int cLength = 1;

    Plato::Evaluator::Expression<RealType> ob(cLength);

    typename Plato::Evaluator::Expression<RealType>::ArrayType tX(cLength, 1.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tY(cLength, 2.0);
    typename Plato::Evaluator::Expression<RealType>::ArrayType tZ(cLength, 3.0);

    ob.set("x", tX);
    ob.set("y", tY);
    ob.set("z", tZ);

    auto tAnswer = ob.evaluate("(x+y)^z");

    auto tAnswer_Host = Kokkos::create_mirror_view(tAnswer);
    Kokkos::deep_copy(tAnswer_Host, tAnswer);

    TEST_FLOATING_EQUALITY(tAnswer_Host[0], 27, 1e-18);
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::Evaluator::Expression

  to compute gold data, execute the following in mathematica:

  D[(x+y)^z,x]/.{x->2,y->2,z->3}

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_sfad_1)
{
    using RealType = Sacado::Fad::SFad<double, 1>;

    constexpr int cLen = 1;

    Plato::Evaluator::Expression<RealType> ob(cLen);

    Plato::ScalarVectorT<RealType> tX("X", cLen);
    Kokkos::deep_copy(tX, RealType(1, 0, 2.0));

    Plato::ScalarVectorT<RealType> tY("Y", cLen);
    Kokkos::deep_copy(tY, 2.0);

    Plato::ScalarVectorT<RealType> tZ("Z", cLen);
    Kokkos::deep_copy(tZ, 3.0);

    ob.set("x", tX);
    ob.set("y", tY);
    ob.set("z", tZ);

    auto tAnswer = ob.evaluate("(x+y)^z");

    auto tAnswer_Host = Kokkos::create_mirror(tAnswer);
    Kokkos::deep_copy(tAnswer_Host, tAnswer);

    for (int i = 0; i < cLen; i++)
    {
        TEST_FLOATING_EQUALITY(tAnswer_Host(i).dx(0), 48, 1e-18);
    }
}
/******************************************************************************/
/*!
  \brief Unit tests for Plato::Evaluator::Expression

  to compute gold data, execute the following in mathematica:

  D[((x1+x2+x3+x4)/4+2)^z,x1]/.{x1->2,x2->2,x3->2,x4->2,y->2,z->3}

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionParser_sfad_4_literals)
{
    using RealType = Sacado::Fad::SFad<double, 4>;

    constexpr int cLen = 1;

    Plato::Evaluator::Expression<RealType> ob(cLen);

    Plato::ScalarVectorT<RealType> tX("X", cLen);
    Kokkos::parallel_for(
        "init", Kokkos::RangePolicy<int>(0, cLen), KOKKOS_LAMBDA(int aOrdinal) {
            tX(aOrdinal) = RealType(4, 0, 2.0) / 4.0;
            for (int j = 1; j < 4; j++)
            {
                tX(aOrdinal) += RealType(4, j, 2.0) / 4.0;
            }
        });

    Plato::ScalarVectorT<RealType> tZ("Z", cLen);
    Kokkos::deep_copy(tZ, 3.0);

    ob.set("x", tX);
    ob.set("z", tZ);

    auto tAnswer = ob.evaluate("(x+2)^z");

    auto tAnswer_Host = Kokkos::create_mirror(tAnswer);
    Kokkos::deep_copy(tAnswer_Host, tAnswer);

    for (int i = 0; i < cLen; i++)
    {
        TEST_FLOATING_EQUALITY(tAnswer_Host(i).dx(0), 12, 1e-18);
        TEST_FLOATING_EQUALITY(tAnswer_Host(i).dx(1), 12, 1e-18);
        TEST_FLOATING_EQUALITY(tAnswer_Host(i).dx(2), 12, 1e-18);
        TEST_FLOATING_EQUALITY(tAnswer_Host(i).dx(3), 12, 1e-18);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Expr1)
{
    std::string tExpr = "-cos(1+Z)";
    auto tFunctor = [](double Z) { return -cos(1 + Z); };

    bool tResult = testExpression<Plato::Scalar>(tExpr, tFunctor);
    TEST_ASSERT(tResult);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LongCeLambda)
{
    std::string tExpr =
        "-2.526344*cos(Z+0.903794)*cos(sin(cos((Z+sin(Z)-92.478617*cos(sin(Z))+85.018428)*sin(Z)+cos(Z)))+0.154748)+0."
        "169827";
    auto tFunctor = [](double Z)
    {
        return -2.526344 * cos(Z + 0.903794) *
                   cos(sin(cos((Z + sin(Z) - 92.478617 * cos(sin(Z)) + 85.018428) * sin(Z) + cos(Z))) + 0.154748) +
               0.169827;
    };

    bool tResult = testExpression<Plato::Scalar>(tExpr, tFunctor);
    TEST_ASSERT(tResult);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortCeLambda)
{
    std::string tExpr = "1e9*((-0.080599*Z-1.559893)*sin(Z-0.376657)*cos((Z+19.353843)*sin(Z-0.376657))-0.336897)";
    auto tFunctor = [](double Z)
    {
        return 1e9 *
               ((-0.080599 * Z - 1.559893) * sin(Z - 0.376657) * cos((Z + 19.353843) * sin(Z - 0.376657)) - 0.336897);
    };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        Plato::Scalar tDeriv = 3.73466541684771252e9;
        using RealType = Plato::Scalar;
        bool tResult = testDerivative<RealType>(tExpr, tDeriv);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortCeMu)
{
    std::string tExpr = "1e9*(-34.60785*Z+37.55432*sin(Z)-0.07816*cos(37.55432*sin(Z))-0.14163)";
    auto tFunctor = [](double Z)
    { return 1e9 * (-34.60785 * Z + 37.55432 * sin(Z) - 0.07816 * cos(37.55432 * sin(Z)) - 0.14163); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        Plato::Scalar tDeriv = -3.57764313783699417e9;
        using RealType = Plato::Scalar;
        bool tResult = testDerivative<RealType>(tExpr, tDeriv);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortCeAlpha)
{
    std::string tExpr = "1e9*(Z*(0.2127003*Z-0.0431028))";
    auto tFunctor = [](double Z) { return 1e9 * (Z * (0.2127003 * Z - 0.0431028)); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortCmLambda)
{
    std::string tExpr = "1e9*(sin(6.4909047*sin(cos(Z)-0.9004168)+0.0669163)-6.6980212*cos(Z)+6.031011)";
    auto tFunctor = [](double Z)
    { return 1e9 * (sin(6.4909047 * sin(cos(Z) - 0.9004168) + 0.0669163) - 6.6980212 * cos(Z) + 6.031011); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortCmMu)
{
    std::string tExpr = "1e9*(0.274743*Z*(Z-0.204501)+0.070614)";
    auto tFunctor = [](double Z) { return 1e9 * (0.274743 * Z * (Z - 0.204501) + 0.070614); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortCmAlpha)
{
    std::string tExpr = "1e9*(sin(Z+0.806286)-0.825645)";
    auto tFunctor = [](double Z) { return 1e9 * (sin(Z + 0.806286) - 0.825645); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortTeLambda)
{
    std::string tExpr = "-0.0764839*Z*(Z-0.4032916)+0.0173456";
    auto tFunctor = [](double Z) { return -0.0764839 * Z * (Z - 0.4032916) + 0.0173456; };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortTeMu)
{
    std::string tExpr = "0.136629*sin(sin(Z+(0.4827465*Z+1)^2*(4.2910311*Z+8.8887868)-16.4881952)-0.9224105)+0.1434269";
    auto tFunctor = [](double Z)
    {
        return 0.136629 *
                   sin(sin(Z + pow(0.4827465 * Z + 1, 2) * (4.2910311 * Z + 8.8887868) - 16.4881952) - 0.9224105) +
               0.1434269;
    };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        Plato::Scalar tDeriv = 1.49638530517222623;
        using RealType = Plato::Scalar;
        bool tResult = testDerivative<RealType>(tExpr, tDeriv);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortTeAlpha)
{
    std::string tExpr = "(1100.1915787767*cos(Z-0.4365025932)-2181.62442608428)*cos(Z-0.4365025932)+1081.5239556353";
    auto tFunctor = [](double Z)
    { return (1100.1915787767 * cos(Z - 0.4365025932) - 2181.62442608428) * cos(Z - 0.4365025932) + 1081.5239556353; };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortJmLambda)
{
    std::string tExpr = "1e9*(-2.6246013*sin(Z*Z+4.392719)-2.6309025)";
    auto tFunctor = [](double Z) { return 1e9 * (-2.6246013 * sin(Z * Z + 4.392719) - 2.6309025); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor, 1e-7);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor, 1e-7);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortJmMu)
{
    std::string tExpr = "1e9*(Z*(0.666224*Z-0.773009)+0.266173)";
    auto tFunctor = [](double Z) { return 1e9 * (Z * (0.666224 * Z - 0.773009) + 0.266173); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortJmAlpha)
{
    std::string tExpr = "1e9*(0.127739*Z*(Z-0.816982)+0.031885)";
    auto tFunctor = [](double Z) { return 1e9 * (0.127739 * Z * (Z - 0.816982) + 0.031885); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortCcMu)
{
    std::string tExpr = "1e9*(0.0515081-0.05260436*sin((Z+101.52609155)*sin(Z+1.01722492)-100.2613725))";
    auto tFunctor = [](double Z)
    { return 1e9 * (0.0515081 - 0.05260436 * sin((Z + 101.52609155) * sin(Z + 1.01722492) - 100.2613725)); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor, 1e-7);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor, 1e-7);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortJcMu)
{
    std::string tExpr = "0.73164851*Z*(Z-2.17317565)+0.73164851*cos(cos(Z*Z*(2*Z-4.3463513))-0.08807638)";
    auto tFunctor = [](double Z)
    { return 0.73164851 * Z * (Z - 2.17317565) + 0.73164851 * cos(cos(Z * Z * (2 * Z - 4.3463513)) - 0.08807638); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ShortTcMu)
{
    std::string tExpr = "0.008536*sin(sin(11.4924668*Z*Z+Z-2.7225632)+1.7967548)+0.0065773";
    auto tFunctor = [](double Z)
    { return 0.008536 * sin(sin(11.4924668 * Z * Z + Z - 2.7225632) + 1.7967548) + 0.0065773; };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Log10)
{
    std::string tExpr = "Log10(10)";
    auto tFunctor = [](double Z) { return log10(10.0); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Log)
{
    std::string tExpr = "Log(Exp(10))";
    auto tFunctor = [](double Z) { return log(exp(10.0)); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
}
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, C11_vs_Temperature)
{
    std::string tExpr = "1.15e10+(7.29e12*Z)/(-2.61e6+Z*Z)";
    auto tFunctor = [](double Z) { return 1.15e10 + (7.29e12 * Z) / (-2.61e6 + Z * Z); };

    {
        using RealType = Plato::Scalar;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        using RealType = Sacado::Fad::SFad<Plato::Scalar, 24>;
        bool tResult = testExpression<RealType>(tExpr, tFunctor);
        TEST_ASSERT(tResult);
    }
    {
        Plato::Scalar tDeriv = -2.79310425089192390e6;
        using RealType = Plato::Scalar;
        bool tResult = testDerivative<RealType>(tExpr, tDeriv);
        TEST_ASSERT(tResult);
    }
}
}  // namespace PlatoUnitTests
