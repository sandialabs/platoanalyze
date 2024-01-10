#pragma once

#include "PlatoTypes.hpp"
#include "ParseTools.hpp"

#include <Kokkos_Core.hpp>
#include <Teuchos_ParameterList.hpp>

#include <cassert>

namespace Plato
{

template<int SpatialDim, typename ScalarType = Plato::Scalar>
class Rank4VoigtConstant
{
public:
    Rank4VoigtConstant() : c0{{0.0}}
    {}

    Rank4VoigtConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
    {}

    KOKKOS_INLINE_FUNCTION ScalarType
    operator()
    (Plato::OrdinalType i, 
     Plato::OrdinalType j ) const
    {
        assert(i < mNumVoigtTerms);
        assert(j < mNumVoigtTerms);
        return c0[i][j];
    }

protected:
    static constexpr auto mNumVoigtTerms     = (SpatialDim == 3) ? 6 :
                                             ((SpatialDim == 2) ? 3 :
                                            (((SpatialDim == 1) ? 1 : 0)));

    ScalarType c0[mNumVoigtTerms][mNumVoigtTerms];

};

template<typename ScalarType>
class Rank4VoigtConstant<1, ScalarType>
{
public:
    Rank4VoigtConstant() : c0{{0.0}} 
    {}

    Rank4VoigtConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
    {
        typedef Plato::Scalar RealT;
        c0[0][0] = Plato::ParseTools::getParam<RealT>(aParams, "c11" /*throw if not found*/);
    }

    KOKKOS_INLINE_FUNCTION ScalarType
    operator()
    (Plato::OrdinalType i, 
     Plato::OrdinalType j ) const 
    {
        assert(i < mNumVoigtTerms);
        assert(j < mNumVoigtTerms);
        return c0[i][j];
    }

protected:
    static constexpr Plato::OrdinalType mNumVoigtTerms = 1;

    ScalarType c0[mNumVoigtTerms][mNumVoigtTerms];
};

template<typename ScalarType>
class Rank4VoigtConstant<2, ScalarType>
{
public:
    Rank4VoigtConstant() : c0{{0.0}} 
    {}

    Rank4VoigtConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
    {
        typedef Plato::Scalar RealT;
        c0[0][0] = Plato::ParseTools::getParam<RealT>(aParams, "c11" /*throw if not found*/);
        c0[1][1] = Plato::ParseTools::getParam<RealT>(aParams, "c22", /*default=*/ c0[0][0]);
        c0[0][1] = Plato::ParseTools::getParam<RealT>(aParams, "c12", /*default=*/ 0.0); c0[1][0] = c0[0][1];
        c0[2][2] = Plato::ParseTools::getParam<RealT>(aParams, "c33" /*throw if not found*/);
    }

    KOKKOS_INLINE_FUNCTION ScalarType
    operator()
    (Plato::OrdinalType i, 
     Plato::OrdinalType j ) const 
    {
        assert(i < mNumVoigtTerms);
        assert(j < mNumVoigtTerms);
        return c0[i][j];
    }

protected:
    static constexpr Plato::OrdinalType mNumVoigtTerms = 3;

    ScalarType c0[mNumVoigtTerms][mNumVoigtTerms];
};

template<typename ScalarType>
class Rank4VoigtConstant<3, ScalarType>
{
public:
    Rank4VoigtConstant() : c0{{0.0}} 
    {}

    Rank4VoigtConstant(Teuchos::ParameterList& aParams) : c0{{0.0}}
    {
        typedef Plato::Scalar RealT;
        c0[0][0] = Plato::ParseTools::getParam<RealT>(aParams, "c11" /*throw if not found*/);
        c0[1][1] = Plato::ParseTools::getParam<RealT>(aParams, "c22", /*default=*/ c0[0][0]);
        c0[2][2] = Plato::ParseTools::getParam<RealT>(aParams, "c33", /*default=*/ c0[0][0]);
        c0[0][1] = Plato::ParseTools::getParam<RealT>(aParams, "c12", /*default=*/ 0.0); c0[1][0] = c0[0][1];
        c0[0][2] = Plato::ParseTools::getParam<RealT>(aParams, "c13", /*default=*/ c0[0][1]); c0[2][0] = c0[0][2];
        c0[1][2] = Plato::ParseTools::getParam<RealT>(aParams, "c23", /*default=*/ c0[0][1]); c0[2][1] = c0[1][2];
        c0[3][3] = Plato::ParseTools::getParam<RealT>(aParams, "c44" /*throw if not found*/);
        c0[4][4] = Plato::ParseTools::getParam<RealT>(aParams, "c55", c0[3][3]);
        c0[5][5] = Plato::ParseTools::getParam<RealT>(aParams, "c66", c0[3][3]);
    }

    KOKKOS_INLINE_FUNCTION ScalarType
    operator()
    (Plato::OrdinalType i, 
     Plato::OrdinalType j ) const 
    {
        assert(i < mNumVoigtTerms);
        assert(j < mNumVoigtTerms);
        return c0[i][j];
    }

protected:
    static constexpr Plato::OrdinalType mNumVoigtTerms = 6;

    ScalarType c0[mNumVoigtTerms][mNumVoigtTerms];
};

}