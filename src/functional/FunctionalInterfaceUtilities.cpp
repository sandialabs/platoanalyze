#include "FunctionalInterfaceUtilities.hpp"

#include <Teuchos_ParameterList.hpp>
#include <boost/functional/hash.hpp>
#include <plato/filter/FilterInterface.hpp>
#include <plato/mesh/MeshDesignVariables.hpp>

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

Teuchos::ParameterList& block_sublist(Teuchos::ParameterList& aParameterList, const std::string_view aBlockName)
{
    return plato_problem_sublist(aParameterList)
        .sublist("Spatial Model")
        .sublist("Domains")
        .sublist(std::string{aBlockName});
}

Teuchos::ParameterList& parameters_sublist(Teuchos::ParameterList& aParameterList)
{
    return plato_problem_sublist(aParameterList).sublist("Parameters");
}

template <typename MeshDesignVariableType, typename Function>
void for_each_design_variable(MeshDesignVariableType&& aDesignVariables,
                              const Plato::Mesh& aMesh,
                              const Function& aFunction)
{
    static_assert(std::is_same_v<std::decay_t<MeshDesignVariableType>, plato::mesh::MeshDesignVariables>);

    const auto& tNodeMap = aMesh->NodeMap();
    for (auto& [tBlockID, tDensityVector] : aDesignVariables.mBlockDensities)
    {
        for (auto& tDensity : tDensityVector)
        {
            const auto tPAControlVectorIndex = tNodeMap.find(tDensity.mGlobalMeshEntityID);
            assert(tPAControlVectorIndex != tNodeMap.cend());
            aFunction(tPAControlVectorIndex->second, tDensity);
        }
    }
}

}  // namespace

[[nodiscard]] std::string first_criterion_name(const Teuchos::ParameterList& aProblem)
{
    const auto tPlatoProblemName = std::string{"Plato Problem"};
    const auto tCriteriaName = std::string{"Criteria"};
    if (aProblem.isSublist(tPlatoProblemName) && aProblem.sublist(tPlatoProblemName).isSublist(tCriteriaName))
    {
        const auto& tCriteriaList = aProblem.sublist(tPlatoProblemName).sublist(tCriteriaName);
        return tCriteriaList.name(tCriteriaList.begin());
    }
    else
    {
        return "";
    }
}

Plato::Comm::Machine create_machine()
{
    MPI_Comm tComm;
    MPI_Comm_dup(MPI_COMM_SELF, &tComm);
    return Plato::Comm::Machine{tComm};
}

Teuchos::ParameterList helmholtz_filter_parameter_list(const filter::library::FilterParameters& aFilterParameters,
                                                       const std::string_view aMeshName,
                                                       const std::vector<std::string>& aBlockNames)
{
    Teuchos::ParameterList tParameterList;
    tParameterList.set("Physics", "Plato Driver");
    tParameterList.set("Spatial Dimension", 3);
    tParameterList.set(std::string{kInputMeshEntry}, std::string{aMeshName});
    plato_problem_sublist(tParameterList).set("Physics", "Helmholtz Filter");
    plato_problem_sublist(tParameterList).set("PDE Constraint", "Helmholtz Filter");
    for (const auto& tBlockName : aBlockNames)
    {
        block_sublist(tParameterList, tBlockName).set("Element Block", tBlockName);
        block_sublist(tParameterList, tBlockName).set("Material Model", "material_1");
    }
    parameters_sublist(tParameterList).set("Length Scale", aFilterParameters.mFilterRadius);
    parameters_sublist(tParameterList)
        .set("Surface Length Scale", aFilterParameters.mBoundaryStickingPenalty.value_or(-1.0));
    return tParameterList;
}

void update_mesh_file_name(Teuchos::ParameterList& aParameterList, const std::string_view aMeshName)
{
    aParameterList.set(std::string{kInputMeshEntry}, std::string{aMeshName});
}

Plato::ScalarVector create_control(const plato::mesh::MeshDesignVariables& aMeshDesignVariables,
                                   const Plato::Mesh& aMesh)
{
    if (aMeshDesignVariables.mBlockDensities.empty())
    {
        Plato::ScalarVector tControl("control", aMesh->NumNodes());
        constexpr double tFullDensity = 1.0;
        Kokkos::deep_copy(tControl, tFullDensity);
        return tControl;
    }
    else
    {
        return full_nodal_scalar_vector(aMeshDesignVariables, aMesh);
    }
}

std::size_t hash_current_design(const Plato::ScalarVector& aControl, const Plato::Mesh& aMesh)
{
    std::size_t tSeed = Plato::detail::hash_vector(aMesh->Coordinates());
    boost::hash_combine(tSeed, Plato::detail::hash_vector(aControl));
    return tSeed;
}

std::vector<double> design_variable_std_vector(const Plato::ScalarVector aScalarVector,
                                               const plato::mesh::MeshDesignVariables& aDesignVariables,
                                               const Plato::Mesh& aMesh)
{
    auto tScalarVectorOnHost = Kokkos::create_mirror_view(aScalarVector);
    Kokkos::deep_copy(tScalarVectorOnHost, aScalarVector);
    auto tDesignVariables = std::vector<double>(number_of_design_variables(aDesignVariables));

    for_each_design_variable(
        aDesignVariables, aMesh,
        [&tDesignVariables, tScalarVectorOnHost](const auto aPAControlIndex, const plato::mesh::Density& aDensity)
        { tDesignVariables[aDensity.mDesignVariableVectorIndex] = tScalarVectorOnHost[aPAControlIndex]; });
    return tDesignVariables;
}

Plato::ScalarVector full_nodal_scalar_vector(const std::vector<double>& aVector,
                                             const plato::mesh::MeshDesignVariables& aDesignVariables,
                                             const Plato::Mesh& aMesh,
                                             const double aFillValue)
{
    const auto tScalarVector = Plato::ScalarVector("control", aMesh->NumNodes());
    auto tScalarVectorOnHost = Kokkos::create_mirror_view(tScalarVector);

    Kokkos::deep_copy(tScalarVectorOnHost, aFillValue);

    for_each_design_variable(
        aDesignVariables, aMesh,
        [&aVector, tScalarVectorOnHost](const auto aPAControlIndex, const plato::mesh::Density& aDensity)
        { tScalarVectorOnHost[aPAControlIndex] = aVector[aDensity.mDesignVariableVectorIndex]; });

    Kokkos::deep_copy(tScalarVector, tScalarVectorOnHost);
    return tScalarVector;
}

Plato::ScalarVector full_nodal_scalar_vector(const plato::mesh::MeshDesignVariables& aDesignVariables,
                                             const Plato::Mesh& aMesh)
{
    const auto& tFirstBlockDensities = aDesignVariables.mBlockDensities.begin()->second;
    const auto tScalarVector = Plato::ScalarVector("control", static_cast<unsigned>(aMesh->NumNodes()));
    auto tScalarVectorOnHost = Kokkos::create_mirror_view(tScalarVector);

    constexpr auto tFixedControlValue = double{1.0};
    Kokkos::deep_copy(tScalarVectorOnHost, tFixedControlValue);

    for_each_design_variable(aDesignVariables, aMesh,
                             [tScalarVectorOnHost](const auto aPAControlIndex, const plato::mesh::Density& aDensity)
                             { tScalarVectorOnHost[aPAControlIndex] = aDensity.mDensity; });

    Kokkos::deep_copy(tScalarVector, tScalarVectorOnHost);
    return tScalarVector;
}

plato::mesh::MeshDesignVariables mesh_design_variables(const Plato::ScalarVector aScalarVector,
                                                       plato::mesh::MeshDesignVariables aDesignVariables,
                                                       const Plato::Mesh& aMesh)
{
    const auto tScalarVectorOnHost = Kokkos::create_mirror_view(aScalarVector);
    Kokkos::deep_copy(tScalarVectorOnHost, aScalarVector);

    for_each_design_variable(aDesignVariables, aMesh,
                             [tScalarVectorOnHost](const auto aPAControlIndex, plato::mesh::Density& aDensity)
                             { aDensity.mDensity = tScalarVectorOnHost[aPAControlIndex]; });

    return aDesignVariables;
}

std::size_t number_of_design_variables(const plato::mesh::MeshDesignVariables& aMeshDesignVariables)
{
    using Density = plato::mesh::Density;
    using IndexType = plato::mesh::Density::IndexType;
    using DensityVector = plato::mesh::MeshDesignVariables::DensityVector;

    if (aMeshDesignVariables.mBlockDensities.empty())
    {
        return 0U;
    }

    const auto tLessVectorIndex = [](const Density& aDensityLeft, const Density& aDensityRight)
    { return aDensityLeft.mDesignVariableVectorIndex < aDensityRight.mDesignVariableVectorIndex; };

    const auto tMaxIndex = [&tLessVectorIndex](const DensityVector& aBlockDensities)
    {
        const auto aMaxIndexIterator =
            std::max_element(aBlockDensities.cbegin(), aBlockDensities.cend(), tLessVectorIndex);
        return aMaxIndexIterator == aBlockDensities.cend() ? IndexType{0}
                                                           : aMaxIndexIterator->mDesignVariableVectorIndex;
    };

    const auto tBlockDensityVector = [&tMaxIndex](const auto& tBlockMapEntry)
    { return tMaxIndex(tBlockMapEntry.second); };

    const auto tMax = [](const IndexType aIndexLeft, const IndexType aIndexRight)
    { return std::max(aIndexLeft, aIndexRight); };

    const auto tMaxVectorIndex =
        std::transform_reduce(aMeshDesignVariables.mBlockDensities.cbegin(),
                              aMeshDesignVariables.mBlockDensities.cend(), IndexType{0}, tMax, tBlockDensityVector);
    return tMaxVectorIndex + 1;
}

}  // namespace plato::functional
