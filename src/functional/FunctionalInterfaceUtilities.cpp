#include "FunctionalInterfaceUtilities.hpp"

#include <Teuchos_ParameterList.hpp>
#include <boost/functional/hash.hpp>

#include <plato/filter/FilterInterface.hpp>
#include <plato/core/MeshProxy.hpp>

#include "CrsMatrixUtils.hpp"
#include "alg/ErrorHandling.hpp"

namespace plato::functional
{
namespace
{
constexpr std::string_view kProblemList = "Problem";
constexpr std::string_view kPlatoProblemList = "Plato Problem";
constexpr std::string_view kInputMeshEntry = "Input Mesh";

Teuchos::ParameterList& plato_problem_sublist(Teuchos::ParameterList& aParameterList)
{
    return aParameterList.sublist(std::string{kPlatoProblemList});
}

Teuchos::ParameterList& block_1_sublist(Teuchos::ParameterList& aParameterList)
{
    return plato_problem_sublist(aParameterList).sublist("Spatial Model").sublist("Domains").sublist("Block 1");
}

Teuchos::ParameterList& parameters_sublist(Teuchos::ParameterList& aParameterList)
{
    return plato_problem_sublist(aParameterList).sublist("Parameters");
}
}  // namespace

Plato::Comm::Machine create_machine()
{
    MPI_Comm tComm = MPI_COMM_WORLD;
    return Plato::Comm::Machine{tComm};
}

Teuchos::ParameterList helmholtz_filter_parameter_list(const filter::library::FilterParameters& aFilterParameters,
                                                       const std::string_view aMeshName)
{
    Teuchos::ParameterList tParameterList;
    tParameterList.set("Physics", "Plato Driver");
    tParameterList.set("Spatial Dimension", 2);  // FIX-ME
    tParameterList.set(std::string{kInputMeshEntry}, std::string{aMeshName});
    plato_problem_sublist(tParameterList).set("Physics", "Helmholtz Filter");
    plato_problem_sublist(tParameterList).set("PDE Constraint", "Helmholtz Filter");
    block_1_sublist(tParameterList).set("Element Block", "block_1");
    block_1_sublist(tParameterList).set("Material Model", "material_1");
    parameters_sublist(tParameterList).set("Length Scale", aFilterParameters.mFilterRadius);
    parameters_sublist(tParameterList)
        .set("Surface Length Scale", aFilterParameters.mBoundaryStickingPenalty.value_or(-1.0));
    return tParameterList;
}

void update_mesh_file_name(Teuchos::ParameterList& aParameterList, const std::string_view aMeshName)
{
    aParameterList.set(std::string{kInputMeshEntry}, std::string{aMeshName});
}

Plato::ScalarVector create_control(const plato::core::MeshProxy& aMeshProxy, const Plato::Mesh& aMesh)
{
    if (aMeshProxy.mNodalDensities.empty())
    {
        Plato::ScalarVector tControl("control", aMesh->NumNodes());
        constexpr double tFullDensity = 1.0;
        Kokkos::deep_copy(tControl, tFullDensity);
        return tControl;
    }
    else
    {
        return to_scalar_vector(aMeshProxy.mNodalDensities);
    }
}

std::size_t hash_current_design(const Plato::ScalarVector& aControl, const Plato::Mesh& aMesh)
{
    std::size_t tSeed = Plato::detail::hash_vector(aMesh->Coordinates());
    boost::hash_combine(tSeed, Plato::detail::hash_vector(aControl));
    return tSeed;
}

std::vector<double> to_std_vector(const Plato::ScalarVector aScalarVector)
{
    auto tScalarVectorOnHost = Kokkos::create_mirror_view(aScalarVector);
    Kokkos::deep_copy(tScalarVectorOnHost, aScalarVector);
    std::vector<double> tResult;
    tResult.reserve(tScalarVectorOnHost.size());
    std::copy(tScalarVectorOnHost.data(), tScalarVectorOnHost.data() + tScalarVectorOnHost.size(),
              std::back_inserter(tResult));
    return tResult;
}

Plato::ScalarVector to_scalar_vector(const std::vector<double>& aVector)
{
    const auto tScalarVector = Plato::ScalarVector("control", aVector.size());
    auto tScalarVectorOnHost = Kokkos::create_mirror_view(tScalarVector);
    for (std::size_t tIndex = 0; tIndex < aVector.size(); ++tIndex)
    {
        tScalarVectorOnHost[tIndex] = aVector[tIndex];
    }
    Kokkos::deep_copy(tScalarVector, tScalarVectorOnHost);
    return tScalarVector;
}
}  // namespace plato::functional
