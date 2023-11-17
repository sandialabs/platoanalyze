#ifndef PLATO_FUNCTIONAL_FUNCTIONALINTERFACE_H
#define PLATO_FUNCTIONAL_FUNCTIONALINTERFACE_H

#include <memory>
#include <utility>

#include <Teuchos_ParameterList.hpp>

#include "alg/ParallelComm.hpp"
#include "PlatoMesh.hpp"

namespace Plato
{
class AbstractProblem;
struct Solutions;

namespace Functional
{
struct MeshProxy;
}
}

namespace Plato::Functional
{
/// @brief The purpose of this class is to provide common functionality to 
/// Plato::Functional interface types.
class FunctionalInterface
{
public:
  FunctionalInterface(Teuchos::ParameterList aParameterList);

  /// @brief Solves the forward problem specified by the ParameterList passed on construction,
  ///  and with an updated mesh @a aMeshProxy.
  auto solveProblem(const MeshProxy& aMeshProxy) 
    -> std::pair<Plato::Solutions, Plato::ScalarVector>;

  Plato::AbstractProblem& problem();
  Teuchos::ParameterList& parameterList();
private:
  Plato::Comm::Machine mMachine;
  Plato::Mesh mMesh;
  std::shared_ptr<Plato::AbstractProblem> mProblem;
  Teuchos::ParameterList mParameterList;
};
}

#endif
