#pragma once

#include <memory>

#include "SimplexFadTypes.hpp"
#include "AnalyzeMacros.hpp"

#include "VonMisesLocalMeasure.hpp"
#include "TensileEnergyDensityLocalMeasure.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

/**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class LocalMeasureFactory
{
/**********************************************************************************/
public:
    LocalMeasureFactory (){}
    ~LocalMeasureFactory (){}
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType, SimplexPhysics>> 
    create(Teuchos::ParameterList& aInputParams, const std::string & aFuncName)
    {
        auto tFunctionSpecs = aInputParams.sublist(aFuncName);
        auto tLocalMeasure = tFunctionSpecs.get<std::string>("Local Measure", "VonMises");

        if(tLocalMeasure == "VonMises")
        {
            return std::make_shared<VonMisesLocalMeasure<EvaluationType, SimplexPhysics>>(aInputParams, "VonMises");
        }
        else if(tLocalMeasure == "TensileEnergyDensity")
        {
            return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType, SimplexPhysics>>
                                                             (aInputParams, "TensileEnergyDensity");
        }
        else
        {
            ANALYZE_THROWERR("Unknown 'Local Measure' specified in 'Plato Problem' ParameterList")
        }
    }
};
// class LocalMeasureFactory

}
//namespace Plato

#include "SimplexMechanics.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::LocalMeasureFactory, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::LocalMeasureFactory, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::LocalMeasureFactory, Plato::SimplexMechanics, 3)
#endif