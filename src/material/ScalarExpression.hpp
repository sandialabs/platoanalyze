#pragma once

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "FadTypes.hpp"
#include "AnalyzeMacros.hpp"
#include "ParseTools.hpp"
#include "utilities/ExpressionParser.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

#include <string>
#include <vector>
#include <string>
#include <map>

namespace Plato
{

template<typename EvaluationType>
class ScalarExpression
{
protected:
    using StateT  = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using KineticsScalarType = typename EvaluationType::ResultScalarType;
    using ElementType = typename EvaluationType::ElementType;
    using KinematicsScalarType = typename Plato::fad_type_t<ElementType, StateT, ConfigT>;
    using ControlScalarType = typename EvaluationType::ControlScalarType;

public:
    ScalarExpression() = default;

    ScalarExpression
    (const std::string& aName, 
     const Teuchos::ParameterList& aParams)
    {
        if (aParams.isSublist(aName)) 
        {
            auto tSubList = aParams.sublist(aName);
            std::vector<std::string> tConstantNames = Plato::ParseTools::getParam<Teuchos::Array<std::string>>(tSubList, "Constant Names").toVector();
            std::vector<Plato::Scalar> tConstantValues = Plato::ParseTools::getParam<Teuchos::Array<Plato::Scalar>>(tSubList, "Constant Values").toVector();
            if(tConstantNames.size() != tConstantValues.size())
            {
                const std::string tErrMessage = "'Constant Names' and 'Constant Values' arrays must have the same number of entries. \n";
                ANALYZE_THROWERR(tErrMessage);
            }
            for(size_t j=0; j<tConstantNames.size(); ++j)
            {
                mConstantsMap[tConstantNames[j]] = tConstantValues[j];
            }
            mIndependentVariableName = Plato::ParseTools::getParam<std::string>(tSubList, "Independent Variable Name", "");
            mStrExpression = Plato::ParseTools::getParam<std::string>(tSubList, "Expression", "");
        }
        else
        { 
            const std::string tErrMessage = "ParameterList for Expression with name '" + aName + "' was not found. \n";
            ANALYZE_THROWERR(tErrMessage);
        }
    }  

    const std::string& getExpression() const { return mStrExpression; }

    const std::map<std::string, Plato::Scalar>& getConstantsMap() const { return mConstantsMap; }

    const std::string& getIndependentVariableName() const { return mIndependentVariableName; }

    Plato::ScalarVectorT<KineticsScalarType>
    operator()(const Plato::ScalarVectorT<ControlScalarType>& aIndependentVariable)  
    {
        mExpression.set(mIndependentVariableName.c_str(), aIndependentVariable);

        std::map<std::string, Plato::Scalar>::iterator tIter = mConstantsMap.begin();
        while(tIter != mConstantsMap.end())
        {
            mExpression.set(tIter->first.c_str(), tIter->second);
            tIter++;
        }

        auto tResult = mExpression.evaluate(mStrExpression);

        // For now, create the return type and convert
        // TODO: this conversion should not be necessary.
        Plato::ScalarVectorT<KineticsScalarType> tReturn("return", tResult.extent(0));
        Kokkos::parallel_for("convert", Kokkos::RangePolicy<Plato::OrdinalType>(0,tResult.extent(0)), KOKKOS_LAMBDA(const Plato::OrdinalType aOrdinal)
        {
          tReturn(aOrdinal) = KineticsScalarType(tResult(aOrdinal));
        });
        return tReturn;

    }

protected:
    std::string mStrExpression;
    std::string mIndependentVariableName;
    std::map<std::string, Plato::Scalar> mConstantsMap;
    Plato::Evaluator::Expression<ControlScalarType> mExpression;
};

}
