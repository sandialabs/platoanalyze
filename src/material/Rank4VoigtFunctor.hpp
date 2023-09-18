#pragma once

#include "PlatoTypes.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

namespace Plato
{

template<int SpatialDim>
class Rank4VoigtFunctor
{
public:
    Rank4VoigtFunctor() : c0{{0.0}}, c1{{0.0}}, c2{{0.0}} {}

    Rank4VoigtFunctor(Teuchos::ParameterList& aParams);

    template<typename TScalarType>
    KOKKOS_INLINE_FUNCTION TScalarType
    operator()
    (TScalarType aInput, 
     Plato::OrdinalType i, 
     Plato::OrdinalType j ) const 
    {
        TScalarType tRetVal(aInput);
        tRetVal *= c1[i][j];
        tRetVal += c0[i][j];
        tRetVal += aInput*aInput*c2[i][j];
        return tRetVal;
    }

protected:
    static constexpr Plato::OrdinalType NumVoigtTerms = (SpatialDim == 3) ? 6 :
                                         ((SpatialDim == 2) ? 3 :
                                        (((SpatialDim == 1) ? 1 : 0)));

    Plato::Scalar c0[NumVoigtTerms][NumVoigtTerms];
    Plato::Scalar c1[NumVoigtTerms][NumVoigtTerms];
    Plato::Scalar c2[NumVoigtTerms][NumVoigtTerms];
};

}