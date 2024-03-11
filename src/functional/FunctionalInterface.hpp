#ifndef PLATO_FUNCTIONAL_FUNCTIONALINTERFACE_H
#define PLATO_FUNCTIONAL_FUNCTIONALINTERFACE_H

#include <Teuchos_ParameterList.hpp>
#include <memory>
#include <utility>

#include "PlatoMesh.hpp"
#include "SolutionCache.hpp"
#include "alg/ParallelComm.hpp"

namespace Plato
{
class AbstractProblem;
struct Solutions;
};  // namespace Plato

namespace plato::core
{
struct MeshProxy;
}

namespace plato::functional
{
/// @brief The purpose of this class is to provide common functionality to
/// plato::functional interface types.
class FunctionalInterface
{
   public:
    FunctionalInterface(Teuchos::ParameterList aParameterList);

    /// @brief Solves the forward problem specified by the ParameterList passed on construction,
    ///  and with an updated mesh @a aMeshProxy.
    auto solveProblem(const core::MeshProxy& aMeshProxy) -> std::pair<Plato::Solutions, Plato::ScalarVector>;

    Plato::AbstractProblem& problem();
    Teuchos::ParameterList& parameterList();

   private:
    Plato::Comm::Machine mMachine;
    Plato::Mesh mMesh;
    std::shared_ptr<Plato::AbstractProblem> mProblem;
    SolutionCache mSolutionCache;
    Teuchos::ParameterList mParameterList;
};
}  // namespace plato::functional

#endif
