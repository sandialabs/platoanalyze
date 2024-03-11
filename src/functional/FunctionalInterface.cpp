#include "FunctionalInterface.hpp"

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <plato/core/MeshProxy.hpp>

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
[[nodiscard]] bool should_update_mesh_dependent_object(const core::MeshProxy& aMeshProxy,
                                                       const std::shared_ptr<T>& aObject)
{
    return !aObject || aMeshProxy.mNodalDensities.empty();
}

/// @brief Updates the mesh @a aMesh with on disk with path found in @a aParameterList if necessary.
///
/// The mesh will only be read from disk if @a aMesh is `nullptr` or @a aMeshProxy does not contain
/// a density vector. A density vector is taken to mean that the mesh is constant and the density field
/// updates the controls.
[[nodiscard]] Plato::Mesh update_mesh(const core::MeshProxy& aMeshProxy, Plato::Mesh&& aMesh)
{
    if (should_update_mesh_dependent_object(aMeshProxy, aMesh))
    {
        return Plato::MeshFactory::create(aMeshProxy.mFileName.string());
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
                                  const core::MeshProxy& aMeshProxy,
                                  const Plato::Mesh& aMesh,
                                  Teuchos::ParameterList& aParameterList,
                                  std::shared_ptr<Plato::AbstractProblem>&& aProblem)
    -> std::shared_ptr<Plato::AbstractProblem>
{
    if (should_update_mesh_dependent_object(aMeshProxy, aProblem))
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
      mSolutionCache{[this](const Plato::ScalarVector& aArg) { return mProblem->solution(aArg); },
                     [this](const Plato::ScalarVector& aArg)
                     { return plato::functional::hash_current_design(aArg, mMesh); }}
{
    start_up();
}

auto FunctionalInterface::solveProblem(const core::MeshProxy& aMeshProxy)
    -> std::pair<Plato::Solutions, Plato::ScalarVector>
{
    Teuchos::ParameterList tParameterList = mParameterList;
    plato::functional::update_mesh_file_name(tParameterList, aMeshProxy.mFileName.string());
    mMesh = update_mesh(aMeshProxy, std::move(mMesh));
    mProblem = update_problem(mMachine, aMeshProxy, mMesh, tParameterList, std::move(mProblem));

    Plato::ScalarVector tControl = plato::functional::create_control(aMeshProxy, mMesh);
    return {mSolutionCache.compute(tControl), tControl};
}

Plato::AbstractProblem& FunctionalInterface::problem() { return *mProblem; }

Teuchos::ParameterList& FunctionalInterface::parameterList() { return mParameterList; }
}  // namespace plato::functional
