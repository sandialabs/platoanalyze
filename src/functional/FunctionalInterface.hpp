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

namespace plato::design_variables
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
    FunctionalInterface();
    explicit FunctionalInterface(Teuchos::ParameterList aParameterList);

    /// @brief Solves the forward problem specified by the ParameterList passed on construction,
    ///  and with an updated mesh given by @a aMeshDesignVariables.
    [[nodiscard]] auto solveProblem(const design_variables::MeshDesignVariables& aMeshDesignVariables)
        -> std::pair<Plato::Solutions, Plato::ScalarVector>;

    /// @brief Solves the forward problem with an updated mesh given by @a aMeshDesignVariables.
    /// @param aMeshDependentParameterListUpdater A function that generates a new ParameterList based on a mesh and @a
    /// aMeshDesignVariables.
    ///  The signature must be `ParameterList(const plato::design_variables::MeshDesignVariables&, const Plato::Mesh&)`.
    template <typename Function>
    [[nodiscard]] auto solveProblem(const design_variables::MeshDesignVariables& aMeshDesignVariables,
                                    const Function& aMeshDependentParameterListUpdater)
        -> std::pair<Plato::Solutions, Plato::ScalarVector>;

    [[nodiscard]] Plato::Solutions computeState(const Plato::ScalarVector& aArg) const;

    [[nodiscard]] Plato::AbstractProblem& problem();
    [[nodiscard]] const Teuchos::ParameterList& parameterList() const;
    [[nodiscard]] const Plato::Mesh& mesh() const;

   private:
    auto updateMesh(const design_variables::MeshDesignVariables& aMeshDesignVariables) -> Teuchos::ParameterList;
    [[nodiscard]] auto solveProblemImpl(const design_variables::MeshDesignVariables& aMeshDesignVariables,
                                        Teuchos::ParameterList aParameterList)
        -> std::pair<Plato::Solutions, Plato::ScalarVector>;

   private:
    Plato::Comm::Machine mMachine;
    Plato::Mesh mMesh;
    std::shared_ptr<Plato::AbstractProblem> mProblem;
    SolutionCache mSolutionCache;
    Teuchos::ParameterList mParameterList;
};

template <typename Function>
auto FunctionalInterface::solveProblem(const design_variables::MeshDesignVariables& aMeshDesignVariables,
                                       const Function& aMeshDependentParameterListUpdater)
    -> std::pair<Plato::Solutions, Plato::ScalarVector>
{
    updateMesh(aMeshDesignVariables);
    return solveProblemImpl(aMeshDesignVariables, aMeshDependentParameterListUpdater(aMeshDesignVariables, mMesh));
}

}  // namespace plato::functional

#endif
