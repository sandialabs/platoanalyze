#ifndef PLATO_FUNCTIONAL_FUNCTIONALINTERFACE_H
#define PLATO_FUNCTIONAL_FUNCTIONALINTERFACE_H

#include <Teuchos_ParameterList.hpp>
#include <memory>
#include <plato/utilities/StateCache.hpp>
#include <utility>

#include "PlatoMesh.hpp"
#include "Solutions.hpp"
#include "alg/ParallelComm.hpp"

namespace Plato
{
class AbstractProblem;
struct Solutions;
};  // namespace Plato

namespace plato::mesh
{
struct MeshDesignVariables;
}

namespace plato::functional
{
/// @brief The purpose of this class is to provide common functionality to
/// plato::functional interface types.
class FunctionalInterface
{
    using SolutionCache = plato::utilities::StateCache<Plato::Solutions, const Plato::ScalarVector&>;

   public:
    FunctionalInterface(Teuchos::ParameterList aParameterList);

    /// @brief Solves the forward problem specified by the ParameterList passed on construction,
    ///  and with an updated mesh @a aMeshProxy.
    auto solveProblem(const mesh::MeshDesignVariables& aMeshDesignVariables)
        -> std::pair<Plato::Solutions, Plato::ScalarVector>;

    Plato::Solutions computeState(const Plato::ScalarVector& aArg) const;

    Plato::AbstractProblem& problem();
    Teuchos::ParameterList& parameterList();
    const Teuchos::ParameterList& parameterList() const;

   private:
    Plato::Comm::Machine mMachine;
    Plato::Mesh mMesh;
    std::shared_ptr<Plato::AbstractProblem> mProblem;
    SolutionCache mSolutionCache;
    Teuchos::ParameterList mParameterList;
};
}  // namespace plato::functional

#endif
