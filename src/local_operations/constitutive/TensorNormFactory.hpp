#ifndef PLATO_LOCAL_OPERATIONS_CONSTITUTIVE_TENSORNORMFACTORY
#define PLATO_LOCAL_OPERATIONS_CONSTITUTIVE_TENSORNORMFACTORY

#include <Teuchos_ParameterList.hpp>
#include <string>

#include "core_types/PlatoTypes.hpp"
#include "local_operations/constitutive/TensorNormBase.hpp"
#include "local_operations/constitutive/VoigtTensorL2Norm.hpp"
#include "local_operations/constitutive/VonMisesNorm.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace Plato
{
template <Plato::OrdinalType VoigtLength, typename EvaluationType>
struct TensorNormFactory
{
    auto create(const Teuchos::ParameterList& aParameterList)
    {
        const auto tExponent = aParameterList.get<Plato::Scalar>("Exponent");
        if (aParameterList.isSublist("Normalize"))
        {
            const auto tNormalizeParameters = aParameterList.sublist("Normalize");
            const auto tNormalizeType = tNormalizeParameters.get<std::string>("Type");
            const bool tNormalizeByVolume = tNormalizeParameters.isType<bool>("Volume Scaling")
                                                ? tNormalizeParameters.get<bool>("Volume Scaling")
                                                : true;
            if (tNormalizeType == "Von Mises")
            {
                return Teuchos::rcp(
                    new TensorNormBase<VoigtLength, EvaluationType>(tExponent, tNormalizeByVolume, VonMisesNorm{}));
            }
            else if (tNormalizeType == "Voigt Tensor L2")
            {
                return Teuchos::rcp(new TensorNormBase<VoigtLength, EvaluationType>(tExponent, tNormalizeByVolume,
                                                                                    VoigtTensorL2Norm{}));
            }
            else
            {
                ANALYZE_THROWERR(
                    std::string("Invalid 'Type' '") + tNormalizeType +
                    std::string("' specified in 'Normalize' sublist of '") + aParameterList.name() +
                    std::string("' parameter list. Valid entries for 'Type' are: 'Von Mises' and 'Voigt Tensor L2'."))
            }
        }
        else
        {
            ANALYZE_THROWERR(
                std::string("Parameter List `") + aParameterList.name() +
                std::string("' is missing a 'Normalize' sublist. This is needed to specify the tensor norm used."));
        }
    }
};

}  // namespace Plato

#endif
