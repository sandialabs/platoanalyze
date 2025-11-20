#include "functional/FunctionalInterface.hpp"

#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <plato/analysis/AnalysisDomainMesh.hpp>

#include "domain/Solutions.hpp"
#include "functional/FunctionalInterfaceUtilities.hpp"
#include "functional/InputValidation.hpp"
#include "functional/ParameterListUtilities.hpp"
#include "main/library/ErrorHandling.hpp"
#include "problem/PlatoAbstractProblem.hpp"
#include "problem/PlatoProblemFactory.hpp"
#include "utilities/ParallelComm.hpp"

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
            auto tThreadsProvided = int{};
            MPI_Init_thread(&tArgc, &tArgv, MPI_THREAD_FUNNELED, &tThreadsProvided);
        }

        if (!Kokkos::is_initialized())
        {
            Kokkos::initialize(tArgc, tArgv);
        }
        Plato::MeshFactory::initialize(tArgc, tArgv);
    }
}

template <typename T>
[[nodiscard]] bool should_update_mesh_dependent_object(const analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                                       const std::shared_ptr<T>& aObject)
{
    return !aObject || aAnalysisDomainMesh.mBlockScalarField.empty();
}

/// @brief Updates the mesh @a aMesh with on disk with path found in @a aParameterList if necessary.
///
/// The mesh will only be read from disk if @a aMesh is `nullptr` or @a aAnalysisDomainMesh does not contain
/// a density vector. A density vector is taken to mean that the mesh is constant and the density field
/// updates the controls.
[[nodiscard]] Plato::Mesh update_mesh(const analysis::AnalysisDomainMesh& aAnalysisDomainMesh, Plato::Mesh&& aMesh)
{
    if (should_update_mesh_dependent_object(aAnalysisDomainMesh, aMesh))
    {
        return Plato::MeshFactory::create(aAnalysisDomainMesh.mFileName.string());
    }
    else
    {
        return std::move(aMesh);
    }
}

/// @brief Updates @a aProblem with the new mesh if necessary.
///
/// The AbstractProblem will only be updated if @a aProblem is `nullptr` or @a aAnalysisDomainMesh does not contain
/// a density vector. A density vector is taken to mean that the mesh is constant and the density field
/// updates the controls.
[[nodiscard]] auto update_problem(Plato::Comm::Machine& aMachine,
                                  const analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                  const Plato::Mesh& aMesh,
                                  Teuchos::ParameterList& aParameterList,
                                  std::shared_ptr<Plato::AbstractProblem>&& aProblem)
    -> std::shared_ptr<Plato::AbstractProblem>
{
    if (should_update_mesh_dependent_object(aAnalysisDomainMesh, aProblem))
    {
        return Plato::ProblemFactory{}.create(aMesh, aParameterList, aMachine);
    }
    else
    {
        return std::move(aProblem);
    }
}
}  // namespace

FunctionalInterface::FunctionalInterface() : FunctionalInterface{Teuchos::ParameterList{}} {}

FunctionalInterface::FunctionalInterface(Teuchos::ParameterList aParameterList)
    : mMachine(plato::functional::create_machine()),
      mParameterList(std::move(aParameterList)),
      mSolutionCache{[this](const Plato::ScalarVector& aArg) { return computeState(aArg); },
                     [this](const Plato::ScalarVector& aArg)
                     { return plato::functional::hash_current_design(aArg, mMesh); }}
{
    start_up();
}

auto FunctionalInterface::solveProblem(const analysis::AnalysisDomainMesh& aAnalysisDomainMesh)
    -> std::pair<Plato::Solutions, Plato::ScalarVector>
{
    auto tUpdatedParameterList = updateMesh(aAnalysisDomainMesh);
    if (const auto tErrorMessage = error_messages(tUpdatedParameterList, mMesh); !tErrorMessage.empty())
    {
        throw std::runtime_error{tErrorMessage};
    }
    return solveProblemImpl(aAnalysisDomainMesh, std::move(tUpdatedParameterList));
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

const Teuchos::ParameterList& FunctionalInterface::parameterList() const { return mParameterList; }

const Plato::Mesh& FunctionalInterface::mesh() const { return mMesh; }

auto FunctionalInterface::updateMesh(const analysis::AnalysisDomainMesh& aAnalysisDomainMesh) -> Teuchos::ParameterList
{
    Teuchos::ParameterList tParameterList = mParameterList;
    plato::functional::update_mesh_file_name(tParameterList, aAnalysisDomainMesh.mFileName.string());
    mMesh = update_mesh(aAnalysisDomainMesh, std::move(mMesh));
    return tParameterList;
}

auto FunctionalInterface::solveProblemImpl(const analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                           Teuchos::ParameterList aParameterList)
    -> std::pair<Plato::Solutions, Plato::ScalarVector>
{
    mProblem = update_problem(mMachine, aAnalysisDomainMesh, mMesh, aParameterList, std::move(mProblem));
    Plato::ScalarVector tControl = plato::functional::create_control(aAnalysisDomainMesh, mMesh);
    return {mSolutionCache.compute(tControl), tControl};
}

}  // namespace plato::functional
