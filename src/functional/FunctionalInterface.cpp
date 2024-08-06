#include "FunctionalInterface.hpp"

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <plato/mesh/MeshDesignVariables.hpp>

#include "FunctionalInterfaceUtilities.hpp"
#include "PlatoAbstractProblem.hpp"
#include "PlatoProblemFactory.hpp"
#include "Solutions.hpp"
#include "alg/ErrorHandling.hpp"
#include "alg/ParallelComm.hpp"

namespace plato::functional
{
namespace
{
/// @brief Initializes services such as mpi and Kokkos.
///
/// This should be called in the ctor of any object interfacing with PlatoFunctional.
void start_up()
{
    static bool tHasStarted = false;
    if (!tHasStarted)
    {
        tHasStarted = true;
        Plato::enable_floating_point_exceptions();

        int tArgc = 0;
        char** tArgv = nullptr;

        int tMPIInitialized = 0;
        MPI_Initialized(&tMPIInitialized);
        if (tMPIInitialized == 0)
        {
            MPI_Init(&tArgc, &tArgv);
        }

        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize(tArgc, tArgv);
        }
        Plato::MeshFactory::initialize(tArgc, tArgv);
    }
}

template <typename T>
[[nodiscard]] bool should_update_mesh_dependent_object(const mesh::MeshDesignVariables& aMeshDesignVariables,
                                                       const std::shared_ptr<T>& aObject)
{
    return !aObject || aMeshDesignVariables.mBlockDensities.empty();
}

/// @brief Updates the mesh @a aMesh with on disk with path found in @a aParameterList if necessary.
///
/// The mesh will only be read from disk if @a aMesh is `nullptr` or @a aMeshDesignVariables does not contain
/// a density vector. A density vector is taken to mean that the mesh is constant and the density field
/// updates the controls.
[[nodiscard]] Plato::Mesh update_mesh(const mesh::MeshDesignVariables& aMeshDesignVariables, Plato::Mesh&& aMesh)
{
    if (should_update_mesh_dependent_object(aMeshDesignVariables, aMesh))
    {
        return Plato::MeshFactory::create(aMeshDesignVariables.mFileName.string());
    }
    else
    {
        return std::move(aMesh);
    }
}

/// @brief Updates @a aProblem with the new mesh if necessary.
///
/// The AbstractProblem will only be updated if @a aProblem is `nullptr` or @a aMeshProxy does not contain
/// a density vector. A density vector is taken to mean that the mesh is constant and the density field
/// updates the controls.
[[nodiscard]] auto update_problem(Plato::Comm::Machine& aMachine,
                                  const mesh::MeshDesignVariables& aMeshDesignVariables,
                                  const Plato::Mesh& aMesh,
                                  Teuchos::ParameterList& aParameterList,
                                  std::shared_ptr<Plato::AbstractProblem>&& aProblem)
    -> std::shared_ptr<Plato::AbstractProblem>
{
    if (should_update_mesh_dependent_object(aMeshDesignVariables, aProblem))
    {
        return Plato::ProblemFactory{}.create(aMesh, aParameterList, aMachine);
    }
    else
    {
        return std::move(aProblem);
    }
}
}  // namespace

FunctionalInterface::FunctionalInterface(Teuchos::ParameterList aParameterList)
    : mMachine(plato::functional::create_machine()),
      mParameterList(std::move(aParameterList)),
      mSolutionCache{[this](const Plato::ScalarVector& aArg) { return computeState(aArg); },
                     [this](const Plato::ScalarVector& aArg)
                     { return plato::functional::hash_current_design(aArg, mMesh); }}
{
    start_up();
}

auto FunctionalInterface::solveProblem(const mesh::MeshDesignVariables& aMeshDesignVariables)
    -> std::pair<Plato::Solutions, Plato::ScalarVector>
{
    Teuchos::ParameterList tParameterList = mParameterList;
    plato::functional::update_mesh_file_name(tParameterList, aMeshDesignVariables.mFileName.string());
    mMesh = update_mesh(aMeshDesignVariables, std::move(mMesh));
    mProblem = update_problem(mMachine, aMeshDesignVariables, mMesh, tParameterList, std::move(mProblem));

    Plato::ScalarVector tControl = plato::functional::create_control(aMeshDesignVariables, mMesh);
    return {mSolutionCache.compute(tControl), tControl};
}

Plato::Solutions FunctionalInterface::computeState(const Plato::ScalarVector& aArg) const
{
    const std::string tCriterionName = first_criterion_name(parameterList());
    if (!tCriterionName.empty() && mProblem->criterionIsLinear(tCriterionName))
    {
        return Plato::Solutions{};
    }
    else
    {
        return mProblem->solution(aArg);
    }
}

Plato::AbstractProblem& FunctionalInterface::problem() { return *mProblem; }

Teuchos::ParameterList& FunctionalInterface::parameterList() { return mParameterList; }

const Teuchos::ParameterList& FunctionalInterface::parameterList() const { return mParameterList; }
}  // namespace plato::functional
