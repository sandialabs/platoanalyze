#pragma once

#include "contact/AbstractContactForce.hpp"
#include "contact/CompliantContactForce.hpp"
#include "contact/NormalContactForce.hpp"

#include "PlatoStaticsTypes.hpp"
#include "PlatoUtilities.hpp"
#include "AnalyzeMacros.hpp"

#include "Teuchos_RCP.hpp"
#include <Teuchos_Array.hpp>

namespace Plato
{

namespace Contact
{

template<typename EvaluationType>
class ContactForceFactory 
{
public:
    ContactForceFactory(){}

    Teuchos::RCP<AbstractContactForce<EvaluationType>>
    create
    (const std::string & aType,
     const Teuchos::Array<Plato::Scalar> & aPenaltyValue)
    {
        auto tType = Plato::tolower(aType);

        if (aType == "tensor")
        {
            return Teuchos::rcp( new CompliantContactForce<EvaluationType>(aPenaltyValue) );
        }
        else if (aType == "normal")
        {
            return Teuchos::rcp( new NormalContactForce<EvaluationType>(aPenaltyValue) );
        }
        else
        {
            ANALYZE_THROWERR("Unknown contact Penalty Type");
        }
    }
};

}

}
