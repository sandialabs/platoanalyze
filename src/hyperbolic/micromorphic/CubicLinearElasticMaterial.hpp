#pragma once

#include "material/CubicStiffnessConstant.hpp"
#include "material/TetragonalSkewStiffnessConstant.hpp"

#include "material/MaterialModel.hpp"
#include "AnalyzeMacros.hpp"

#include <string>

namespace Plato::Hyperbolic::Micromorphic
{

template<Plato::OrdinalType SpatialDim>
class CubicLinearElasticMaterial : public MaterialModel<SpatialDim>
{
    enum class MatrixType { Voigt, Skew };

public:
    CubicLinearElasticMaterial(const Teuchos::ParameterList& aParamList)
    {
        this->setRank4Tensor(aParamList, "Ce", MatrixType::Voigt);
        this->setRank4Tensor(aParamList, "Cc", MatrixType::Skew);
        this->setRank4Tensor(aParamList, "Cm", MatrixType::Voigt);
        this->setModelType();
    }

private:
    void
    setRank4Tensor
    (const Teuchos::ParameterList& aParamList,
     const std::string& aName,
     const MatrixType aType)
    {
        const std::string tListName = aName + " Stiffness Tensor";
        const std::string tExpressionName = tListName + " Expression";
        if(aParamList.isSublist(tListName))
        {
            auto tParams = aParamList.sublist(tListName);
            if(aType == MatrixType::Voigt)
                this->setRank4VoigtConstant(aName, Plato::CubicStiffnessConstant<SpatialDim>(tParams));
            else if(aType == MatrixType::Skew)
                this->setRank4SkewConstant(aName, Plato::TetragonalSkewStiffnessConstant<SpatialDim>(tParams));
        }
        else if (aParamList.isSublist(tExpressionName))
        {
            this->parseRank4Field(tExpressionName, aParamList, aName);
        }
        else
            ANALYZE_THROWERR("ParameterList for Cubic Micromorphic Elastic Material must have name " + tListName + " or " + tExpressionName)
    }

};

}