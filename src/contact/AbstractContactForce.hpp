#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Contact
{

template<typename EvaluationType>
class AbstractContactForce : public EvaluationType::ElementType
{
protected:
    using StateType  = typename EvaluationType::StateScalarType;  
    using ConfigType = typename EvaluationType::ConfigScalarType; 
    using ResultType = typename EvaluationType::ResultScalarType; 

public:
    AbstractContactForce() = default;

    virtual ~AbstractContactForce() = default;

    AbstractContactForce(const AbstractContactForce& aForce) = delete;

    AbstractContactForce(AbstractContactForce&& aForce) = delete;

    AbstractContactForce&
    operator=(const AbstractContactForce& aForce) = delete;

    AbstractContactForce&
    operator=(AbstractContactForce&& aForce) = delete;

    virtual void
    operator()
    (const Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds,
     const Plato::OrdinalVectorT<const Plato::OrdinalType> & aLocalNodeOrds,
     const Plato::ScalarArray3DT<StateType>  & aState,
     const Plato::ScalarArray3DT<ConfigType> & aConfig,
           Plato::ScalarArray3DT<ResultType> & aResult) const = 0;
};

}

}
