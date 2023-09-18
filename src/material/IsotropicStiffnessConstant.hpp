#pragma once

#include "material/Rank4VoigtConstant.hpp"

#include "PlatoTypes.hpp"
#include "ParseTools.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

template<int SpatialDim, typename ScalarType = Plato::Scalar>
class IsotropicStiffnessConstant : public Rank4VoigtConstant<SpatialDim, ScalarType>
{
public:
    IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams);
};

template<typename ScalarType>
class IsotropicStiffnessConstant<1, ScalarType> : public Rank4VoigtConstant<1, ScalarType>
{
public:
    IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams)
    {
        typedef Plato::Scalar RealT;
        const ScalarType E = Plato::ParseTools::getParam<RealT>(aParams, "Youngs Modulus"); /*throw if not found*/
        const ScalarType v = Plato::ParseTools::getParam<RealT>(aParams, "Poissons Ratio"); /*throw if not found*/
        const ScalarType c = 1.0/((1.0+v)*(1.0-2.0*v));

        this->c0[0][0] = E*c*(1.0-v);
    }
};

template<typename ScalarType>
class IsotropicStiffnessConstant<2, ScalarType> : public Rank4VoigtConstant<2, ScalarType>
{
public:
    IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams)
    {
        typedef Plato::Scalar RealT;
        const ScalarType E = Plato::ParseTools::getParam<RealT>(aParams, "Youngs Modulus"); /*throw if not found*/
        const ScalarType v = Plato::ParseTools::getParam<RealT>(aParams, "Poissons Ratio"); /*throw if not found*/

        const ScalarType c = 1.0/((1.0+v)*(1.0-2.0*v));

        const ScalarType c00 = E*c*(1.0-v), c01 = E*c*v, c22 = 1.0/2.0*E*c*(1.0-2.0*v);

        this->c0[0][0] = c00; 
        this->c0[0][1] = c01;
        this->c0[1][0] = c01; 
        this->c0[1][1] = c00;
        this->c0[2][2] = c22;
    }
};

template<typename ScalarType>
class IsotropicStiffnessConstant<3, ScalarType> : public Rank4VoigtConstant<3, ScalarType>
{
public:
    IsotropicStiffnessConstant(const Teuchos::ParameterList& aParams)
    {
        typedef Plato::Scalar RealT;
        const ScalarType E = Plato::ParseTools::getParam<RealT>(aParams, "Youngs Modulus"); /*throw if not found*/
        const ScalarType v = Plato::ParseTools::getParam<RealT>(aParams, "Poissons Ratio"); /*throw if not found*/

        const ScalarType c = 1.0/((1.0+v)*(1.0-2.0*v));

        const ScalarType c00 = E*c*(1.0-v), c01 = E*c*v, c33 = 1.0/2.0*E*c*(1.0-2.0*v);

        this->c0[0][0] = c00; 
        this->c0[0][1] = c01; 
        this->c0[0][2] = c01;
        this->c0[1][0] = c01; 
        this->c0[1][1] = c00; 
        this->c0[1][2] = c01;
        this->c0[2][0] = c01; 
        this->c0[2][1] = c01; 
        this->c0[2][2] = c00;
        this->c0[3][3] = c33; 
        this->c0[4][4] = c33; 
        this->c0[5][5] = c33;
    }
};

}