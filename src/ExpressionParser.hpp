#pragma once
/******************************************************************

This is a modified version of math expression parser presented in 
the book : "C++ The Complete Reference" by H.Schildt.

-- supports operators: + - * / ^ ( )

-- supports math functions : SIN, COS, TAN, ASIN, ACOS, ATAN, SINH, 
COSH, TANH, ASINH, ACOSH, ATANH, LN, LOG, EXP, SQRT, SQR.

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

template <typename Real = double, typename Int = int>
class RealArray
{
  public:

  using data_type = Plato::ScalarVectorT<Real>;
  using array_type = RealArray<Real,Int>;

  data_type mData;

  // Constructors
  RealArray() {}
  explicit RealArray(Int aLength, Real aInit=0.0)
  {
    mData = data_type("data", aLength);
    Kokkos::deep_copy(mData, aInit);
  }
  RealArray(array_type const & aArray)
  {
    mData = data_type("data", aArray.mData.extent(0));
    Kokkos::deep_copy(mData, aArray.mData);
  }

  RealArray(data_type const & aData)
  {
    mData = aData;
  }

  // Unary operators
  array_type operator- () {
    array_type tArray(*this);
    auto tA = tArray.mData;
    Kokkos::parallel_for("operator-", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) = -tA(aOrdinal);
    });
    return tArray;
  }

  array_type operator+ () {
   return array_type(*this);
  }

  // binary operators 
  array_type operator+(const array_type& b) {
    array_type tArray(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator+", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) += tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator-(const array_type& b) {
    array_type tArray(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator-", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) -= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator*(const array_type& b) {
    array_type tArray(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator*", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) *= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator/(const array_type& b) {
    array_type tArray(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator/", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) /= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator=(const Real& b) {
    array_type tArray(*this);
    auto tA = tArray.mData;
    Kokkos::deep_copy(tA, b);
    return tArray;
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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
  RealArray<Real,Int> tArray(aArray);
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


enum types { DELIMITER = 1, VARIABLE, NUMBER, FUNCTION };

template <typename RealType, typename IntType=int>
class Expression {

  public: 
  using ArrayType = RealArray<RealType, IntType>;

  private:
  IntType mVectorLength;
  char *mExpression;
  char mToken[256];
  char mTokenType;
  std::map<std::string,ArrayType> mVariables;

  void assignment(ArrayType &aResult)
  {
    char tTempToken[80];
    if (mTokenType == VARIABLE)
    {
      // save old token
      char *tPtr = mExpression;
      strcpy(tTempToken, mToken);
      std::string tKey = mToken;
      // compute the index of the variable
      getToken();
      if (*mToken != '=')
      {
        mExpression = tPtr; // return current token
        strcpy(mToken, tTempToken); // restore old token
        mTokenType = VARIABLE;
      }
      else {
        getToken(); // get next part of exp
        add_subtract(aResult);
        mVariables[tKey] = aResult;
        return;
      }
    }
    add_subtract(aResult);
  }


  void add_subtract(ArrayType &aResult)
  {
    char op;
    ArrayType tTemp;
    multiply_divide(aResult);
    while ((op = *mToken) == '+' || op == '-')
    {
      getToken();
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
    char op;
    ArrayType tTemp;
    exponent(aResult);
    while ((op = *mToken) == '*' || op == '/')
    {
      getToken();
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
    while (*mToken == '^')
    {
      getToken();
      unary_plus_minus(tTemp);
      aResult = Math::pow(aResult, tTemp);
    }
  }

  void unary_plus_minus(ArrayType &aResult)
  {
    char op;
    op = 0;
    if ((mTokenType == DELIMITER) && *mToken == '+' || *mToken == '-')
    {
      op = *mToken;
      getToken();
    }
    process(aResult);
    if (op == '-')
      aResult = -aResult;
  }

  // Process a function, a parenthesized expression, a value or a variable
  void process(ArrayType &aResult)
  {
    bool tIsFunction = (mTokenType == FUNCTION);
    char tTempToken[80];
    if (tIsFunction)
    {
      strcpy(tTempToken, mToken);
      getToken();
    } 
    if ((*mToken == '(')) 
    {
      getToken();
      add_subtract(aResult);
      if (*mToken != ')')
        ANALYZE_THROWERR("Evaluator: Unbalanced Parentheses");
      if (tIsFunction)
      {
        if (!strcasecmp(tTempToken, "SIN"))
          aResult = Math::sin(aResult);
        else if (!strcasecmp(tTempToken, "COS"))
          aResult = Math::cos(aResult);
        else if (!strcasecmp(tTempToken, "TAN"))
          aResult = Math::tan(aResult);
        else if (!strcasecmp(tTempToken, "ASIN"))
          aResult = Math::asin(aResult);
        else if (!strcasecmp(tTempToken, "ACOS"))
          aResult = Math::acos(aResult);
        else if (!strcasecmp(tTempToken, "ATAN"))
          aResult = Math::atan(aResult);
        else if (!strcasecmp(tTempToken, "SINH"))
          aResult = Math::sinh(aResult);
        else if (!strcasecmp(tTempToken, "COSH"))
          aResult = Math::cosh(aResult);
        else if (!strcasecmp(tTempToken, "TANH"))
          aResult = Math::tanh(aResult);
        else if (!strcasecmp(tTempToken, "ASINH"))
          aResult = Math::asinh(aResult);
        else if (!strcasecmp(tTempToken, "ACOSH"))
          aResult = Math::acosh(aResult);
        else if (!strcasecmp(tTempToken, "ATANH"))
          aResult = Math::atanh(aResult);
        else if (!strcasecmp(tTempToken, "LN"))
          aResult = Math::log(aResult);
        else if (!strcasecmp(tTempToken, "LOG"))
          aResult = Math::log10(aResult);
        else if (!strcasecmp(tTempToken, "EXP"))
          aResult = Math::exp(aResult);
        else if (!strcasecmp(tTempToken, "SQRT"))
          aResult = Math::sqrt(aResult);
        else if (!strcasecmp(tTempToken, "SQR"))
          aResult = aResult*aResult;
        else
        {
          std::stringstream error;
          error << "Expression contains an unknown function: " << tTempToken;
          ANALYZE_THROWERR("Unknown Function");
        }
      }
      getToken();
    }
    else 
    {
      if (mTokenType == VARIABLE)
      {
        std::string tKey = mToken;
        if(mVariables.count(tKey) == 0)
        {
          std::stringstream error;
          error << "Expression contains undefined variable: " << tKey;
          ANALYZE_THROWERR(error.str())
        }
        aResult = mVariables.at(tKey);
        getToken();
        return;
      } else
      if (mTokenType == NUMBER)
      {
        aResult = ArrayType(mVectorLength,atof(mToken));
        getToken();
        return;
      } else
      {
        ANALYZE_THROWERR("Evaluator: Syntax Error");
      }
    }
  }


  // 1. set mToken to the next token
  // 2. set mTokenType to the token type
  // 3. advance mExpression past the next token
  void getToken()
  {
    char *tTemp;
    mTokenType = 0;
    tTemp = mToken;
    *tTemp = '\0';
    if (!*mExpression)  // at end of expression
      return;
    while (isspace(*mExpression))  // skip over white space
      ++mExpression; 
    if (strchr("+-*/%^=()", *mExpression)) 
    {
      mTokenType = DELIMITER;
      *tTemp++ = *mExpression++;  // advance to next char
    }
    else if (isalpha(*mExpression)) 
    {
      while (!strchr(" +-/*%^=()\t\r", *mExpression) && (*mExpression))
        *tTemp++ = *mExpression++;
      while (isspace(*mExpression))  // skip over white space
        ++mExpression;
      mTokenType = (*mExpression == '(') ? FUNCTION : VARIABLE;
    }
    else if (isdigit(*mExpression) || *mExpression == '.')
    {
      while (!strchr(" +-/*%^=()\t\r", *mExpression) && (*mExpression))
        *tTemp++ = toupper(*mExpression++);
      mTokenType = NUMBER;
    }
    *tTemp = '\0';
  }

  public:

  Expression(IntType aVectorLength=0) :
    mVectorLength(aVectorLength),
    mExpression(NULL) {}

  void set(std::string aName, Plato::Scalar aValue)
  {
    mVariables[aName] = ArrayType(mVectorLength,aValue);
  }

  void set(std::string aName, ArrayType aValue)
  {
    if (mVectorLength == 0)
    {
      mVectorLength = aValue.mData.extent(0);
    }
    assert(mVectorLength == aValue.mData.extent(0));

    mVariables[aName] = aValue;
  }

  typename ArrayType::data_type get(std::string aName)
  {
    return mVariables[aName].mData;
  }

  typename ArrayType::data_type evaluate(std::string aExpression)
  {
    ArrayType tResult;
    mExpression = new char [aExpression.length()+1];
    std::strcpy (mExpression, aExpression.c_str());

    getToken();

    if (!*mToken)
    {
      ANALYZE_THROWERR("Evaluator called with an empty expression.");
    }

    assignment(tResult);

    if (*mToken) // last token must be null
      ANALYZE_THROWERR("Evaluator: Syntax Error");
    return tResult.mData;
  }

};

} // end namespace Parser

} // end namespace Plato
