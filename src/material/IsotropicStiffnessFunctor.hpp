#pragma once

#include <Teuchos_ParameterList.hpp>

#include "material/Rank4VoigtFunctor.hpp"

namespace Plato
{

template <int SpatialDim>
class IsotropicStiffnessFunctor : public Rank4VoigtFunctor<SpatialDim>
{
   public:
    IsotropicStiffnessFunctor(const Teuchos::ParameterList& aParams);
};

}  // namespace Plato
