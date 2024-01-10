#include "ScalarFunctor.hpp"

#include "PlatoTypes.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Kokkos_Core.hpp>

namespace Plato
{

ScalarFunctor::ScalarFunctor() : c0(0.0), c1(0.0), c2(0.0) {}

ScalarFunctor::ScalarFunctor(Plato::Scalar aVal) : c0(aVal), c1(0.0), c2(0.0) {}

ScalarFunctor::ScalarFunctor(Teuchos::ParameterList& aParams) : c0(0.0), c1(0.0), c2(0.0)
{
    if (aParams.isType<Plato::Scalar>("c0"))
    {
        c0 = aParams.get<Plato::Scalar>("c0");
    }
    else
    {
        ANALYZE_THROWERR("Missing required parameter 'c0'");
    }

    if (aParams.isType<Plato::Scalar>("c1"))
    {
        c1 = aParams.get<Plato::Scalar>("c1");
    }

    if (aParams.isType<Plato::Scalar>("c2"))
    {
        c2 = aParams.get<Plato::Scalar>("c2");
    }
}

}