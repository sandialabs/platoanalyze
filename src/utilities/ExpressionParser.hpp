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

#include <math.h>

#include <Sacado.hpp>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#include "PlatoStaticsTypes.hpp"
#include "PlatoUtilities.hpp"
#include "utilities/RealArray.hpp"

namespace Plato
{

namespace Evaluator
{

enum class TokenType
{
    Delimiter,
    Variable,
    Number,
    Function
};

template <typename RealType, typename IntType = int>
class Expression
{
   public:
    using ArrayType = RealArray<RealType, IntType>;
    using StringType = std::string;
    using CharType = StringType::value_type;

   private:
    IntType mVectorLength;
    StringType mToken;
    CharType *mExpression;
    TokenType mTokenType;
    std::map<StringType, ArrayType> mVariables;
    std::map<const StringType, const std::function<ArrayType(ArrayType)>> mFunctions = {
        {"sin", Math::sin<RealType, IntType>},     {"cos", Math::cos<RealType, IntType>},
        {"tan", Math::tan<RealType, IntType>},     {"asin", Math::asin<RealType, IntType>},
        {"acos", Math::acos<RealType, IntType>},   {"atan", Math::atan<RealType, IntType>},
        {"sinh", Math::sinh<RealType, IntType>},   {"cosh", Math::cosh<RealType, IntType>},
        {"tanh", Math::tanh<RealType, IntType>},   {"asinh", Math::asinh<RealType, IntType>},
        {"acosh", Math::acosh<RealType, IntType>}, {"atanh", Math::atanh<RealType, IntType>},
        {"log", Math::log<RealType, IntType>},     {"log10", Math::log10<RealType, IntType>},
        {"exp", Math::exp<RealType, IntType>},     {"sqrt", Math::sqrt<RealType, IntType>},
        {"sqr", Math::sqr<RealType, IntType>}};

    void assignment(ArrayType &aResult)
    {
        StringType tToken;
        if (mTokenType == TokenType::Variable)
        {
            // save old token
            auto tPtr = mExpression;
            tToken = mToken;
            // compute the index of the variable
            advanceTokenAndExpression();
            if (mToken.front() != '=')
            {
                mExpression = tPtr;  // return current token
                mToken = tToken;     // restore old token
                mTokenType = TokenType::Variable;
            }
            else
            {
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
        while ((op = mToken.front()) == '+' || op == '-')
        {
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
        while ((op = mToken.front()) == '*' || op == '/')
        {
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
        while (mToken.front() == '^')
        {
            advanceTokenAndExpression();
            unary_plus_minus(tTemp);
            aResult = Math::pow(aResult, tTemp);
        }
    }

    void unary_plus_minus(ArrayType &aResult)
    {
        CharType op(0);
        if ((mTokenType == TokenType::Delimiter) && mToken.front() == '+' || mToken.front() == '-')
        {
            op = mToken.front();
            advanceTokenAndExpression();
        }
        process(aResult);
        if (op == '-') aResult = -aResult;
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
        if (mToken.front() == '(')
        {
            advanceTokenAndExpression();
            add_subtract(aResult);
            if (mToken.front() != ')') ANALYZE_THROWERR("Evaluator: Unbalanced Parentheses");
            if (tIsFunction)
            {
                auto tTokenLower = Plato::tolower(tToken);
                if (mFunctions.count(tTokenLower))
                {
                    aResult = mFunctions.at(tTokenLower)(aResult);
                }
                else
                {
                    std::stringstream error;
                    error << "Expression contains an unknown function: " << tToken;
                    ANALYZE_THROWERR(error.str());
                }
            }
            advanceTokenAndExpression();
        }
        else
        {
            if (mTokenType == TokenType::Variable)
            {
                StringType tKey = mToken;
                if (mVariables.count(tKey) == 0)
                {
                    std::stringstream error;
                    error << "Expression contains undefined variable: " << tKey;
                    ANALYZE_THROWERR(error.str())
                }
                aResult = mVariables.at(tKey);
                advanceTokenAndExpression();
                return;
            }
            else if (mTokenType == TokenType::Number)
            {
                aResult = ArrayType(mVectorLength, atof(mToken.data()));
                advanceTokenAndExpression();
                return;
            }
            else
            {
                ANALYZE_THROWERR("Evaluator: Syntax Error");
            }
        }
    }

    // 1. set mToken to the next token
    // 2. set mTokenType to the token type
    // 3. advance mExpression past the next token
    void advanceTokenAndExpression()
    {
        // char *tTemp;
        mTokenType = TokenType::Delimiter;
        mToken.clear();
        if (!*mExpression)  // at end of expression
            return;
        while (isspace(*mExpression))  // skip over white space
            ++mExpression;
        if (strchr("+-*/^=()", *mExpression))
        {
            mTokenType = TokenType::Delimiter;
            mToken += *mExpression++;  // advance to next char
        }
        else if (isalpha(*mExpression))
        {
            while (!strchr(" +-/*^=()\t\r", *mExpression) && (*mExpression)) mToken += *mExpression++;
            while (isspace(*mExpression))  // skip over white space
                ++mExpression;
            mTokenType = (*mExpression == '(') ? TokenType::Function : TokenType::Variable;
        }
        else if (isdigit(*mExpression) || *mExpression == '.')
        {
            while (!strchr(" +-/*^=()\t\r", *mExpression) && (*mExpression)) mToken += toupper(*mExpression++);
            mTokenType = TokenType::Number;
        }
    }

   public:
    Expression(IntType aVectorLength = 0) : mVectorLength(aVectorLength), mExpression(nullptr) {}

    void set(const StringType &aName, Plato::Scalar aValue) { mVariables[aName] = ArrayType(mVectorLength, aValue); }

    void set(const StringType &aName, const ArrayType &aValue)
    {
        if (mVectorLength == 0)
        {
            mVectorLength = aValue.mData.extent(0);
        }
        assert(mVectorLength == aValue.mData.extent(0));

        mVariables[aName] = Math::copy(aValue);
    }

    typename ArrayType::data_type get(const StringType &aName) { return mVariables[aName].mData; }

    [[maybe_unused]] typename ArrayType::data_type evaluate(StringType aExpression)
    {
        mExpression = aExpression.data();

        advanceTokenAndExpression();

        if (mToken.empty())
        {
            ANALYZE_THROWERR("Evaluator called with an empty expression.");
        }

        ArrayType tResult;
        assignment(tResult);

        if (mToken.empty() == false)  // last token must be null
            ANALYZE_THROWERR("Evaluator: Syntax Error");

        return tResult.mData;
    }
};

}  // namespace Evaluator

}  // end namespace Plato
