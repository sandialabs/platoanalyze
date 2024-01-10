#include "TensorConstant.hpp"

#include "ParseTools.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

template<>
TensorConstant<1>::TensorConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
{
    typedef Plato::Scalar T;
    c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c11" /*throw if not found*/);

}

template<>
TensorConstant<2>::TensorConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
{
    typedef Plato::Scalar T;
    c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c11" /*throw if not found*/);
    c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c22", /*default=*/ c0[0][0]);
    c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c12", /*default=*/ 0.0); c0[1][0] = c0[0][1];
}

template<>
TensorConstant<3>::TensorConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
{
    typedef Plato::Scalar T;
    c0[0][0] = Plato::ParseTools::getParam<T>(aParams, "c11" /*throw if not found*/);
    c0[1][1] = Plato::ParseTools::getParam<T>(aParams, "c22", /*default=*/ c0[0][0]);
    c0[0][1] = Plato::ParseTools::getParam<T>(aParams, "c12", /*default=*/ 0.0); c0[1][0] = c0[0][1];
    c0[2][2] = Plato::ParseTools::getParam<T>(aParams, "c33", /*default=*/ c0[0][0]);
    c0[0][2] = Plato::ParseTools::getParam<T>(aParams, "c13", /*default=*/ 0.0); c0[2][0] = c0[0][2];
    c0[1][2] = Plato::ParseTools::getParam<T>(aParams, "c23", /*default=*/ 0.0); c0[2][1] = c0[1][2];
}

}