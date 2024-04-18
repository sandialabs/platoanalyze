#pragma once
/******************************************************************

This is a modified version of math expression parser presented in
the book : "C++ The Complete Reference" by H.Schildt.

-- supports operators: + - * / ^ ( )

-- supports math functions : SIN, COS, TAN, ASIN, ACOS, ATAN, SINH,
COSH, TANH, ASINH, ACOSH, ATANH, LOG (natural), LOG10, EXP, SQRT, SQR.

-- supports case-sensitive variables, and can evaluate a sequence
of expressions.  See the unit tests below for use cases.

-- built on the Plato::ScalarVectorT type to permit use of AD types
and to ensure that data remain in device memory during evaluation.

*******************************************************************/

#include <iostream>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <string>
#include <math.h> 

#include <Sacado.hpp>
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Evaluator
{

namespace Math
{

template <typename T>
T copy(const T& aArray)
{
  T tArray;
  tArray.mData = typename T::data_type("data", aArray.mData.extent(0));
  Kokkos::deep_copy(tArray.mData, aArray.mData);
  return tArray;
}

} // end namespace Math


template <typename Real = double, typename Int = int>
class RealArray
{
  public:

  using data_type = Plato::ScalarVectorT<Real>;
  using array_type = RealArray<Real,Int>;

  data_type mData;

  // Constructors
  RealArray()=default;
  RealArray(array_type &&)=delete;
  RealArray(array_type const &)=default;
  array_type& operator=(array_type&&)=default;
  array_type& operator=(const array_type&)=default;

  explicit RealArray(Int aLength, Real aInit=0.0)
  {
    mData = data_type("data", aLength);
    Kokkos::deep_copy(mData, aInit);
  }

  RealArray(data_type const & aData)
  {
    mData = aData;
  }

  // Unary operators
  array_type operator- () {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    Kokkos::parallel_for("operator-", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) = -tA(aOrdinal);
    });
    return tArray;
  }

  array_type operator+ () {
   return Math::copy(*this);
  }

  // binary operators 
  array_type operator+(const array_type& b) {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator+", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) += tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator-(const array_type& b) {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator-", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) -= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator*(const array_type& b) {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator*", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) *= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator/(const array_type& b) {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator/", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) /= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator=(const Real& b) {
    Kokkos::deep_copy(mData, b);
    return *this;
  }

  // I/O
  template <typename fReal, typename fInt>
  friend
  std::ostream& operator<<(std::ostream& os, const RealArray<fReal,fInt>& aArray);

};

namespace Math
{

template <typename Real, typename Int>
RealArray<Real,Int> pow(const RealArray<Real,Int>& aArray, double aExp)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("pow", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::pow(tB(aOrdinal), aExp);
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> pow(const RealArray<Real,Int>& aArray, const RealArray<Real,Int>& aExp)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  auto tC = aExp.mData;
  Kokkos::parallel_for("pow", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::pow(tB(aOrdinal), tC(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> sin(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("sin", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::sin(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> cos(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("cos", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::cos(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> tan(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("tan", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::tan(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> asin(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("asin", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::asin(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> acos(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("acos", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::acos(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> atan(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("atan", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::atan(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> sinh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("sinh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::sinh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> cosh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("cosh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::cosh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> tanh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("tanh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::tanh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> asinh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("asinh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::asinh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> acosh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("acosh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::acosh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> atanh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("atanh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::atanh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> log(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("log", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::log(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> log10(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("log10", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::log10(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> exp(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("exp", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::exp(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> sqrt(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("sqrt", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::sqrt(tB(aOrdinal));
  });
  return tArray;
}

}

template <typename Real = double, typename Int = int>
std::ostream& operator<<(std::ostream& os, const RealArray<Real,Int>& aArray)
{
    auto tData_Host = Kokkos::create_mirror_view(aArray.mData);
    Kokkos::deep_copy(tData_Host, aArray.mData);
    for (const Real& v : tData_Host)
    {
      os << " " << v << " ";
    }
    return os;
}

enum class TokenType { Delimiter, Variable, Number, Function };

template <typename RealType, typename IntType=int>
class Expression {

  public: 
  using ArrayType = RealArray<RealType, IntType>;
  using StringType = std::string;
  using CharType = StringType::value_type;

 private:
  IntType mVectorLength;
  StringType mToken;
  CharType* mExpression;
  TokenType mTokenType;
  std::map<StringType, ArrayType> mVariables;

  void assignment(ArrayType &aResult)
  {
    StringType tToken;
    if (mTokenType == TokenType::Variable) {
      // save old token
      auto tPtr = mExpression;
      tToken = mToken;
      // compute the index of the variable
      advanceTokenAndExpression();
      if (mToken.front() != '=') {
        mExpression = tPtr; // return current token
        mToken = tToken;    // restore old token
        mTokenType = TokenType::Variable;
      } else {
        advanceTokenAndExpression();  // get next part of exp
        add_subtract(aResult);
        mVariables[tToken] = aResult;
        return;
      }
    }
    add_subtract(aResult);
  }


  void add_subtract(ArrayType &aResult)
  {
    CharType op;
    ArrayType tTemp;
    multiply_divide(aResult);
    while ((op = mToken.front()) == '+' || op == '-') {
      advanceTokenAndExpression();
      multiply_divide(tTemp);
      switch (op)
      {
      case '-':
        aResult = aResult - tTemp;
        break;
      case '+':
        aResult = aResult + tTemp;
        break;
      }
    }
  }

  void multiply_divide(ArrayType &aResult)
  {
    CharType op;
    ArrayType tTemp;
    exponent(aResult);
    while ((op = mToken.front()) == '*' || op == '/') {
      advanceTokenAndExpression();
      exponent(tTemp);
      switch (op)
      {
      case '*':
        aResult = aResult * tTemp;
        break;
      case '/':
        aResult = aResult / tTemp;
        break;
      }
    }
  }

  void exponent(ArrayType &aResult)
  {
    ArrayType tTemp;
    unary_plus_minus(aResult);
    while (mToken.front() == '^') {
      advanceTokenAndExpression();
      unary_plus_minus(tTemp);
      aResult = Math::pow(aResult, tTemp);
    }
  }

  void unary_plus_minus(ArrayType &aResult)
  {
    CharType op(0);
    if ((mTokenType == TokenType::Delimiter) && mToken.front() == '+' || mToken.front() == '-') {
      op = mToken.front();
      advanceTokenAndExpression();
    }
    process(aResult);
    if (op == '-')
      aResult = -aResult;
  }

  // Process a function, a parenthesized expression, a value or a variable
  void process(ArrayType &aResult)
  {
    auto tIsFunction = (mTokenType == TokenType::Function);
    StringType tToken;
    if (tIsFunction)
    {
      tToken = mToken;
      advanceTokenAndExpression();
    }
    if ((mToken.front() == '(')) {
      advanceTokenAndExpression();
      add_subtract(aResult);
      if (mToken.front() != ')') ANALYZE_THROWERR("Evaluator: Unbalanced Parentheses");
      if (tIsFunction)
      {
        if (!strcasecmp(tToken.data(), "SIN"))
          aResult = Math::sin(aResult);
        else if (!strcasecmp(tToken.data(), "COS"))
          aResult = Math::cos(aResult);
        else if (!strcasecmp(tToken.data(), "TAN"))
          aResult = Math::tan(aResult);
        else if (!strcasecmp(tToken.data(), "ASIN"))
          aResult = Math::asin(aResult);
        else if (!strcasecmp(tToken.data(), "ACOS"))
          aResult = Math::acos(aResult);
        else if (!strcasecmp(tToken.data(), "ATAN"))
          aResult = Math::atan(aResult);
        else if (!strcasecmp(tToken.data(), "SINH"))
          aResult = Math::sinh(aResult);
        else if (!strcasecmp(tToken.data(), "COSH"))
          aResult = Math::cosh(aResult);
        else if (!strcasecmp(tToken.data(), "TANH"))
          aResult = Math::tanh(aResult);
        else if (!strcasecmp(tToken.data(), "ASINH"))
          aResult = Math::asinh(aResult);
        else if (!strcasecmp(tToken.data(), "ACOSH"))
          aResult = Math::acosh(aResult);
        else if (!strcasecmp(tToken.data(), "ATANH"))
          aResult = Math::atanh(aResult);
        else if (!strcasecmp(tToken.data(), "LOG"))
          aResult = Math::log(aResult);
        else if (!strcasecmp(tToken.data(), "LOG10"))
          aResult = Math::log10(aResult);
        else if (!strcasecmp(tToken.data(), "EXP"))
          aResult = Math::exp(aResult);
        else if (!strcasecmp(tToken.data(), "SQRT"))
          aResult = Math::sqrt(aResult);
        else if (!strcasecmp(tToken.data(), "SQR"))
          aResult = aResult*aResult;
        else
        {
          std::stringstream error;
          error << "Expression contains an unknown function: " << tToken;
          ANALYZE_THROWERR("Unknown Function");
        }
      }
      advanceTokenAndExpression();
    } else {
      if (mTokenType == TokenType::Variable) {
        StringType tKey = mToken;
        if(mVariables.count(tKey) == 0)
        {
          std::stringstream error;
          error << "Expression contains undefined variable: " << tKey;
          ANALYZE_THROWERR(error.str())
        }
        aResult = mVariables.at(tKey);
        advanceTokenAndExpression();
        return;
      } else if (mTokenType == TokenType::Number) {
        aResult = ArrayType(mVectorLength, atof(mToken.data()));
        advanceTokenAndExpression();
        return;
      } else {
        ANALYZE_THROWERR("Evaluator: Syntax Error");
      }
    }
  }


  // 1. set mToken to the next token
  // 2. set mTokenType to the token type
  // 3. advance mExpression past the next token
  void advanceTokenAndExpression() {
    // char *tTemp;
    mTokenType = TokenType::Delimiter;
    mToken.clear();
    if (!*mExpression)  // at end of expression
      return;
    while (isspace(*mExpression))  // skip over white space
      ++mExpression;
    if (strchr("+-*/^=()", *mExpression)) {
      mTokenType = TokenType::Delimiter;
      mToken += *mExpression++;  // advance to next char
    } else if (isalpha(*mExpression)) {
      while (!strchr(" +-/*^=()\t\r", *mExpression) && (*mExpression)) mToken += *mExpression++;
      while (isspace(*mExpression))  // skip over white space
        ++mExpression;
      mTokenType = (*mExpression == '(') ? TokenType::Function : TokenType::Variable;
    } else if (isdigit(*mExpression) || *mExpression == '.') {
      while (!strchr(" +-/*^=()\t\r", *mExpression) && (*mExpression)) mToken += toupper(*mExpression++);
      mTokenType = TokenType::Number;
    }
  }

  public:

  Expression(IntType aVectorLength=0) :
    mVectorLength(aVectorLength),
    mExpression(nullptr) {}

  void set(const StringType& aName, Plato::Scalar aValue) { mVariables[aName] = ArrayType(mVectorLength, aValue); }

  void set(const StringType& aName, const ArrayType& aValue) {
    if (mVectorLength == 0)
    {
      mVectorLength = aValue.mData.extent(0);
    }
    assert(mVectorLength == aValue.mData.extent(0));

    mVariables[aName] = Math::copy(aValue);
  }

  typename ArrayType::data_type get(const StringType& aName) { return mVariables[aName].mData; }

  [[maybe_unused]] typename ArrayType::data_type evaluate(StringType aExpression) {
    mExpression = aExpression.data();

    advanceTokenAndExpression();

    if (mToken.empty()) {
      ANALYZE_THROWERR("Evaluator called with an empty expression.");
    }

    ArrayType tResult;
    assignment(tResult);

    if (mToken.empty() == false)  // last token must be null
      ANALYZE_THROWERR("Evaluator: Syntax Error");

    return tResult.mData;
  }
};

} // end namespace Parser

} // end namespace Plato
