#pragma once

#include <Teuchos_Array.hpp>
#include <Teuchos_RCP.hpp>

#include "domain/contact/AbstractContactForce.hpp"
#include "domain/contact/CompliantContactForce.hpp"
#include "domain/contact/NormalContactForce.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "utilities/AnalyzeMacros.hpp"
#include "utilities/PlatoUtilities.hpp"

namespace Plato
{

namespace Contact
{

template <typename EvaluationType>
class ContactForceFactory
{
   public:
    ContactForceFactory() {}

    Teuchos::RCP<AbstractContactForce<EvaluationType>> create(const std::string& aType,
                                                              const Teuchos::Array<Plato::Scalar>& aPenaltyValue) const
    {
        auto tType = Plato::tolower(aType);

        if (aType == "tensor")
        {
            return Teuchos::rcp(new CompliantContactForce<EvaluationType>(aPenaltyValue));
        }
        else if (aType == "normal")
        {
            return Teuchos::rcp(new NormalContactForce<EvaluationType>(aPenaltyValue));
        }
        else
        {
            ANALYZE_THROWERR("Unknown contact Penalty Type");
        }
    }
};

}  // namespace Contact

}  // namespace Plato
