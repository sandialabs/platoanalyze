#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
class TrapezoidIntegrator
/******************************************************************************/
{
    Plato::Scalar mAlpha;

public:
    /******************************************************************************/
    explicit 
    TrapezoidIntegrator(Teuchos::ParameterList& aParams) :
      mAlpha( aParams.get<double>("Trapezoid Alpha", /*default=*/ 0.5) )
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    Plato::Scalar inline
    v_grad_u( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return -1.0/(mAlpha*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar inline
    v_grad_u_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return 1.0/(mAlpha*aTimeStep);
    }

    /******************************************************************************/
    Plato::Scalar inline
    v_grad_v_prev( Plato::Scalar aTimeStep )
    /******************************************************************************/
    {
        return (1.0-mAlpha)/mAlpha;
    }


    /******************************************************************************/
    Plato::ScalarVector inline
    v_value(const Plato::ScalarVector & aU,
            const Plato::ScalarVector & aU_prev,
            const Plato::ScalarVector & aV,
            const Plato::ScalarVector & aV_prev,
                  Plato::Scalar dt)
    /******************************************************************************/
    {
        auto tNumData = aU.extent(0);
        Plato::ScalarVector tReturnValue("velocity residual", tNumData);

        auto tAlpha = mAlpha;
        Kokkos::parallel_for("Velocity residual value", Kokkos::RangePolicy<>(0, tNumData), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
        {
            tReturnValue(aOrdinal) = aV(aOrdinal) - 1.0/(tAlpha*dt)*(aU(aOrdinal) - aU_prev(aOrdinal) - dt*(1.0-tAlpha)*aV_prev(aOrdinal));
        });

        return tReturnValue;
    }
};

} // namespace Parabolic

} // namespace Plato
