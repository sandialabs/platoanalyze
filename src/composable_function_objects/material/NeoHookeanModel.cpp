#include "composable_function_objects/material/NeoHookeanModel.hpp"

#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"

namespace plato::composable_function_objects::material
{
NeoHookeanParameters get_neo_hookean_parameters(const Teuchos::ParameterList& aMaterialParamList)
{
    if (aMaterialParamList.isSublist("Neo Hookean Hyperelastic"))
    {
        NeoHookeanParameters tParameters{};
        tParameters.mBulkModulus =
            aMaterialParamList.sublist("Neo Hookean Hyperelastic").get<Plato::Scalar>("Bulk Modulus");
        tParameters.mShearModulus =
            aMaterialParamList.sublist("Neo Hookean Hyperelastic").get<Plato::Scalar>("Shear Modulus");
        return tParameters;
    }
    else
    {
        ANALYZE_THROWERR("Material model contains no Neo Hookean material parameters.");
    }
}

NeoHookeanModel::NeoHookeanModel(const NeoHookeanParameters& aParameters) : mParameters{aParameters} {}

}  // namespace plato::composable_function_objects::material
