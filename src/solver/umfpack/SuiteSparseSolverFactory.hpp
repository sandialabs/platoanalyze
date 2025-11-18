#pragma once

#include <memory>

#include "solver/PlatoAbstractSolver.hpp"

namespace Teuchos
{
class ParameterList;
}

namespace Plato::alg
{
/// @brief Constructs either CHOLMODLinearSolver or UMFPACKLinearSolver based on the type of linear system given by @a
/// aLinearSystemType.
[[nodiscard]] auto make_suite_sparse_solver(const Teuchos::ParameterList& aSolverParams,
                                            Plato::LinearSystemType aLinearSystemType,
                                            std::shared_ptr<Plato::MultipointConstraints> aMPCs = {})
    -> std::unique_ptr<AbstractSolver>;
}  // namespace Plato::alg
