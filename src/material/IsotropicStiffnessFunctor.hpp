#pragma once

#include "material/Rank4VoigtFunctor.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

template<int SpatialDim>
class IsotropicStiffnessFunctor : public Rank4VoigtFunctor<SpatialDim>
{
public:
    IsotropicStiffnessFunctor(const Teuchos::ParameterList& aParams);

};

}