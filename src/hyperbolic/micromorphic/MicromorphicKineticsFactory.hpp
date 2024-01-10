#pragma once

#include "AbstractMicromorphicKinetics.hpp"
#include "LinearMicromorphicKinetics.hpp"
#include "ExpressionMicromorphicKinetics.hpp"

#include "material/MaterialModel.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_RCP.hpp>

namespace Plato::Hyperbolic::Micromorphic
{

template<typename EvaluationType, typename ElementType>
class MicromorphicKineticsFactory
{
public:
    MicromorphicKineticsFactory() = default;

    Teuchos::RCP<AbstractMicromorphicKinetics<EvaluationType, ElementType>> 
    create(const Teuchos::RCP<Plato::MaterialModel<ElementType::mNumSpatialDims>> aMaterialModel)
    {
        Plato::MaterialModelType tModelType = aMaterialModel->type();
        if (tModelType == Plato::MaterialModelType::Linear)
        {
            return Teuchos::rcp( new LinearMicromorphicKinetics<EvaluationType, ElementType>(aMaterialModel) );
        }
        else if (tModelType == Plato::MaterialModelType::Expression)
        {
            return Teuchos::rcp( new ExpressionMicromorphicKinetics<EvaluationType, ElementType>(aMaterialModel) );
        }
        else
        {
            ANALYZE_THROWERR("Unknown Material Model Type in MicromorphicKineticsFactory")
        }
    }

};

}