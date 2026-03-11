#include "solver/PlatoSolverFactory.hpp"

#include "solver/amgx/AmgXLinearSolver.hpp"
#include "solver/tpetra/TpetraLinearSolver.hpp"
#include "utilities/PlatoUtilities.hpp"

#ifdef PLATO_TACHO
#include "solver/tacho/TachoLinearSolver.hpp"
#endif
#ifdef PLATO_UMFPACK
#include "solver/umfpack/SuiteSparseSolverFactory.hpp"
#endif

namespace Plato
{

std::string determine_solver_stack(const Teuchos::ParameterList& tSolverParams)
{
    std::string tSolverStack;
    if (tSolverParams.isType<std::string>("Solver Stack"))
    {
        tSolverStack = tSolverParams.get<std::string>("Solver Stack");
    }
    else
    {
#ifdef PLATO_UMFPACK
        tSolverStack = "UMFPACK";
#elif PLATO_TACHO
        tSolverStack = "Tacho";
#elif HAVE_AMGX
        tSolverStack = "AmgX";
#else
        tSolverStack = "Tpetra";
#endif
    }

    return tSolverStack;
}

/******************************************************************************/
/**
 * @brief Solver factory for AbstractSolvers with MPCs
 **********************************************************************************/
rcp<AbstractSolver> SolverFactory::create(Plato::OrdinalType aNumNodes,
                                          Comm::Machine aMachine,
                                          Plato::OrdinalType aDofsPerNode,
                                          std::shared_ptr<Plato::MultipointConstraints> aMPCs)
{
    auto tSolverStack = Plato::determine_solver_stack(mSolverParams);
    auto tLowerSolverStack = Plato::tolower(tSolverStack);

    if (tLowerSolverStack == "tpetra")
    {
        const Plato::OrdinalType tNumCondensedNodes = (aMPCs == nullptr) ? aNumNodes : aMPCs->getNumCondensedNodes();
        return std::make_shared<Plato::TpetraLinearSolver>(mSolverParams, tNumCondensedNodes, aMachine, aDofsPerNode,
                                                           aMPCs);
    }
    else if (tLowerSolverStack == "amgx")
    {
#ifdef HAVE_AMGX
        return std::make_shared<Plato::AmgXLinearSolver>(mSolverParams, aDofsPerNode, aMPCs);
#else
        ANALYZE_THROWERR("Not compiled with AmgX");
#endif
    }
    else if (tLowerSolverStack == "tacho")
    {
#ifdef PLATO_TACHO
        return std::make_shared<tacho::TachoLinearSolver>(mSolverParams, mType, aMPCs);
#else
        ANALYZE_THROWERR("Not compiled with Tacho");
#endif
    }
    else if (tLowerSolverStack == "umfpack")
    {
#ifdef PLATO_UMFPACK
        return std::shared_ptr<AbstractSolver>{alg::make_suite_sparse_solver(mSolverParams, mType, aMPCs)};
#else
        ANALYZE_THROWERR("Not compiled with UMFPACK");
#endif
    }
    ANALYZE_THROWERR("Requested solver stack not found");
}

}  // end namespace Plato
