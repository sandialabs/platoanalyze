#pragma once

#include <Kokkos_Core.hpp>
#include <Teuchos_ParameterList.hpp>

#include "core_types/PlatoTypes.hpp"

namespace Plato
{

class ScalarFunctor
{
   public:
    ScalarFunctor();

    ScalarFunctor(Plato::Scalar aVal);

    ScalarFunctor(Teuchos::ParameterList& aParams);

    template <typename TScalarType>
    KOKKOS_INLINE_FUNCTION TScalarType operator()(TScalarType aInput) const
    {
        TScalarType tRetVal(aInput);
        tRetVal *= c1;
        tRetVal += c0;
        tRetVal += aInput * aInput * c2;
        return tRetVal;
    }

   private:
    Plato::Scalar c0, c1, c2;
};

}  // namespace Plato
