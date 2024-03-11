#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Contact
{

template <typename EvaluationType>
class AbstractSurfaceDisplacement
{
protected:
    using InStateT  = typename EvaluationType::StateScalarType;  
    using OutStateT   = typename EvaluationType::StateScalarType; 

public:
    AbstractSurfaceDisplacement(Plato::Scalar aScale = 1.0) : mScale(aScale) 
    {}

    virtual ~AbstractSurfaceDisplacement() = default;

    AbstractSurfaceDisplacement(const AbstractSurfaceDisplacement& aDisp) = delete;

    AbstractSurfaceDisplacement(AbstractSurfaceDisplacement&& aDisp) = delete;

    AbstractSurfaceDisplacement&
    operator=(const AbstractSurfaceDisplacement& aDisp) = delete;

    AbstractSurfaceDisplacement&
    operator=(AbstractSurfaceDisplacement&& aDisp) = delete;

    virtual void
    operator()
    (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds,
     const Plato::ScalarMultiVectorT<InStateT>             & aState,
           Plato::ScalarArray3DT<OutStateT>                & aSurfaceDisp) const = 0;

protected:
    Plato::Scalar mScale;
};

}

}