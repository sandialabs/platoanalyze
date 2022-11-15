#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Contact
{

template <typename EvaluationType>
class AbstractSurfaceDisplacement : public EvaluationType::ElementType
{
protected:
    using ElementType = typename EvaluationType::ElementType;
    using InStateT  = typename EvaluationType::StateScalarType;  
    using OutStateT = typename EvaluationType::ResultScalarType; 

public:
    AbstractSurfaceDisplacement(Plato::Scalar aScale = 1.0) : mScale(aScale) 
    {}

    virtual ~AbstractSurfaceDisplacement(){}

    virtual KOKKOS_INLINE_FUNCTION void
    operator()
    (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds,
     const Plato::ScalarMultiVectorT<InStateT>             & aState,
           Plato::ScalarMultiVectorT<OutStateT>            & aSurfaceDisp) const = 0;

protected:
    Plato::Scalar mScale;
};

}

}