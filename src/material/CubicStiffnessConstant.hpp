#pragma once

#include "material/MaterialModel.hpp"
#include "PlatoTypes.hpp"
#include "ParseTools.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

template<int SpatialDim, typename ScalarType = Plato::Scalar>
class CubicStiffnessConstant : public Rank4VoigtConstant<SpatialDim, ScalarType>
{
public:
    CubicStiffnessConstant(const Teuchos::ParameterList& aParams);
};

template<typename ScalarType>
class CubicStiffnessConstant<1, ScalarType> : public Rank4VoigtConstant<1, ScalarType>
{
public:
    CubicStiffnessConstant(const Teuchos::ParameterList& aParams) 
    {
        const ScalarType tLambda = Plato::ParseTools::getParam<double>(aParams, "Lambda"); 
        const ScalarType tMu = Plato::ParseTools::getParam<double>(aParams, "Mu"); 

        this->c0[0][0] = tLambda + 2.0 * tMu;
    }
};

template<typename ScalarType>
class CubicStiffnessConstant<2, ScalarType> : public Rank4VoigtConstant<2, ScalarType>
{
public:
    CubicStiffnessConstant(const Teuchos::ParameterList& aParams) 
    {
        const ScalarType tLambda = Plato::ParseTools::getParam<double>(aParams, "Lambda"); 
        const ScalarType tMu = Plato::ParseTools::getParam<double>(aParams, "Mu"); 
        const ScalarType tAlpha = Plato::ParseTools::getParam<double>(aParams, "Alpha"); 

        this->c0[0][0] = tLambda + 2.0 * tMu; 
        this->c0[0][1] = tLambda;
        this->c0[1][0] = tLambda; 
        this->c0[1][1] = tLambda + 2.0 * tMu;
        this->c0[2][2] = tAlpha;
    }
};

template<typename ScalarType>
class CubicStiffnessConstant<3, ScalarType> : public Rank4VoigtConstant<3, ScalarType>
{
public:
    CubicStiffnessConstant(const Teuchos::ParameterList& aParams) 
    {
        const ScalarType tLambda = Plato::ParseTools::getParam<double>(aParams, "Lambda"); 
        const ScalarType tMu = Plato::ParseTools::getParam<double>(aParams, "Mu"); 
        const ScalarType tAlpha = Plato::ParseTools::getParam<double>(aParams, "Alpha"); 

        this->c0[0][0] = tLambda + 2.0 * tMu; 
        this->c0[0][1] = tLambda; 
        this->c0[0][2] = tLambda;
        this->c0[1][0] = tLambda; 
        this->c0[1][1] = tLambda + 2.0 * tMu; 
        this->c0[1][2] = tLambda;
        this->c0[2][0] = tLambda; 
        this->c0[2][1] = tLambda; 
        this->c0[2][2] = tLambda + 2.0 * tMu;
        this->c0[3][3] = tAlpha; 
        this->c0[4][4] = tAlpha; 
        this->c0[5][5] = tAlpha;
    }
};

}