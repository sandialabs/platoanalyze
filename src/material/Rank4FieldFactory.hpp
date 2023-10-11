#pragma once

#include "material/Rank4Field.hpp"
#include "material/IsotropicVoigtRank4Field.hpp"
#include "material/CubicVoigtRank4Field.hpp"
#include "material/TetragonalSkewRank4Field.hpp"

#include "AnalyzeMacros.hpp"
#include "ParseTools.hpp"

#include <Teuchos_ParameterList.hpp>

#include <memory>
#include <string>

namespace Plato
{

template<int SpaceDim>
class Rank4FieldFactory
{
public:
    Rank4FieldFactory() = default;

    Rank4FieldFactory(const Teuchos::ParameterList& aParams) :
    mParams(aParams)
    {}

    template<typename EvaluationType>
    std::shared_ptr<Rank4Field<EvaluationType>>
    create() const
    {
        const std::string tSymmetry = Plato::ParseTools::getParam<std::string>(mParams, "symmetry");
        if(tSymmetry == "isotropic")
        {
            return std::make_shared<IsotropicVoigtRank4Field<EvaluationType>>(mParams);
        }
        else if(tSymmetry == "cubic")
        {
            return std::make_shared<CubicVoigtRank4Field<EvaluationType>>(mParams);
        }
        else if(tSymmetry == "tetragonal skew")
        {
            return std::make_shared<TetragonalSkewRank4Field<EvaluationType>>(mParams);
        }
        else
        {
            const std::string tErrMessage = "Attempted to create material with non-supported symmetry " + tSymmetry + "\n";
            ANALYZE_THROWERR(tErrMessage);
        }

    }

private:
    Teuchos::ParameterList mParams;
};

}