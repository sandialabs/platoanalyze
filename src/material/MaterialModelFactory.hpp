#pragma once

#include "material/MaterialModel.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>

#include <string>

namespace Plato
{

template<int SpatialDim>
class MaterialModelFactory
{
public:

    MaterialModelFactory(const Teuchos::ParameterList& aParamList) :
    mParamList(aParamList)
    {}

    virtual ~MaterialModelFactory() = default;

    MaterialModelFactory(const MaterialModelFactory& aFactory) = delete;

    MaterialModelFactory(MaterialModelFactory&& aFactory) = delete;

    MaterialModelFactory&
    operator=(const MaterialModelFactory& aFactory) = delete;

    MaterialModelFactory&
    operator=(MaterialModelFactory&& aFactory) = delete;

    Teuchos::RCP<Plato::MaterialModel<SpatialDim>>
    create(const std::string& aModelName)
    {
        if (!mParamList.isSublist("Material Models"))
        {
            REPORT("'Material Models' list not found! Returning 'nullptr'");
            return Teuchos::RCP<Plato::MaterialModel<SpatialDim>>(nullptr);
        }
        else
        {
            auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");

            if (!tModelsParamList.isSublist(aModelName))
            {
                const std::string tErrMessage = "Requested a material model ('" + aModelName + "') that isn't defined \n";
                ANALYZE_THROWERR(tErrMessage);
            }

            auto tModelParamList = tModelsParamList.sublist(aModelName);

            return this->constructFromSublist(tModelParamList);
        }
    }

protected:
    virtual Teuchos::RCP<Plato::MaterialModel<SpatialDim>>
    constructFromSublist(const Teuchos::ParameterList& aParamList) = 0;

private:
    const Teuchos::ParameterList& mParamList;

};

}