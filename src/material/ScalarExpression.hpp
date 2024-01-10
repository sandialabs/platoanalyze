#pragma once

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "FadTypes.hpp"
#include "AnalyzeMacros.hpp"
#include "ParseTools.hpp"
#include "ExpressionEvaluator.hpp"

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
            mExpression = Plato::ParseTools::getParam<std::string>(tSubList, "Expression", "");
        }
        else
        { 
            const std::string tErrMessage = "ParameterList for Expression with name '" + aName + "' was not found. \n";
            ANALYZE_THROWERR(tErrMessage);
        }
    }  

    const std::string& getExpression() const { return mExpression; }

    const std::map<std::string, Plato::Scalar>& getConstantsMap() const { return mConstantsMap; }

    const std::string& getIndependentVariableName() const { return mIndependentVariableName; }

    Plato::ScalarMultiVectorT<KineticsScalarType>
    operator()(const Plato::ScalarVectorT<ControlScalarType>& aIndependentVariable)  
    {
        // aIndependentVariable is of dimension (numCells*numCubaturePointsPerCell, 1)
        auto tCubWeights = ElementType::getCubWeights();
        Plato::OrdinalType tNumPoints = tCubWeights.size();
        Plato::OrdinalType tNumCells = aIndependentVariable.size() / tNumPoints;

        mExpEval.parse_expression(mExpression.c_str());
        mExpEval.setup_storage(tNumCells*tNumPoints, 1);
        std::map<std::string, Plato::Scalar>::iterator tIter = mConstantsMap.begin();
        while(tIter != mConstantsMap.end())
        {
            mExpEval.set_variable(tIter->first.c_str(), tIter->second);
            tIter++;
        }
        mExpEval.set_variable(mIndependentVariableName.c_str(), aIndependentVariable);

        auto &tExpEval = mExpEval;
        Plato::ScalarMultiVectorT<KineticsScalarType> tResults("Expression Results", tNumCells*tNumPoints, 1);

        Kokkos::parallel_for("compute element values", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType iCellOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tEntryOrdinal = iCellOrdinal*tNumPoints + iGpOrdinal;
            tExpEval.evaluate_expression( tEntryOrdinal, tResults );
        });

        return tResults;
    }

protected:
    std::string mExpression;
    std::string mIndependentVariableName;
    std::map<std::string, Plato::Scalar> mConstantsMap;
    ExpressionEvaluator<Plato::ScalarMultiVectorT<KineticsScalarType>,
                        Plato::ScalarMultiVectorT<KinematicsScalarType>,
                        Plato::ScalarVectorT<ControlScalarType>,
                        Plato::Scalar > mExpEval;

};

}