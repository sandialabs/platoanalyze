#pragma once

#include <Teuchos_ParameterList.hpp>

#include "ParseTools.hpp"
#include "material/Rank4SkewConstant.hpp"

namespace Plato
{

template <int SpatialDim, typename ScalarType = Plato::Scalar>
class TetragonalSkewStiffnessConstant : public Rank4SkewConstant<SpatialDim, ScalarType>
{
   public:
    TetragonalSkewStiffnessConstant(const Teuchos::ParameterList& aParams) {}
};

template <typename ScalarType>
class TetragonalSkewStiffnessConstant<1, ScalarType> : public Rank4SkewConstant<1, ScalarType>
{
   public:
    TetragonalSkewStiffnessConstant(const Teuchos::ParameterList& aParams)
    {
        ScalarType tMu = Plato::ParseTools::getParam<double>(aParams, "Mu");
        this->c0[0][0] = tMu;
    }
};

template <typename ScalarType>
class TetragonalSkewStiffnessConstant<2, ScalarType> : public Rank4SkewConstant<2, ScalarType>
{
   public:
    TetragonalSkewStiffnessConstant(const Teuchos::ParameterList& aParams)
    {
        ScalarType tMu = Plato::ParseTools::getParam<double>(aParams, "Mu");
        this->c0[0][0] = tMu;
    }
};

template <typename ScalarType>
class TetragonalSkewStiffnessConstant<3, ScalarType> : public Rank4SkewConstant<3, ScalarType>
{
   public:
    TetragonalSkewStiffnessConstant(const Teuchos::ParameterList& aParams)
    {
        ScalarType tMu = Plato::ParseTools::getParam<double>(aParams, "Mu");

        this->c0[0][0] = tMu;
        this->c0[1][1] = tMu;
        this->c0[2][2] = tMu;
    }
};

}  // namespace Plato
