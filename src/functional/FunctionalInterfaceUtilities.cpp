#include "FunctionalInterfaceUtilities.hpp"

#include <Kokkos_StdAlgorithms.hpp>
#include <Teuchos_ParameterList.hpp>
#include <boost/functional/hash.hpp>
#include <plato/analysis/AnalysisDomainMesh.hpp>
#include <plato/filter/library/FilterInterface.hpp>

#include "CrsMatrixUtils.hpp"
#include "ParameterListUtilities.hpp"
#include "alg/ErrorHandling.hpp"

namespace plato::functional
{
namespace
{
constexpr auto kInputMeshEntry = std::string_view{"Input Mesh"};
constexpr auto kHelmholtzFilterName = std::string_view{"Helmholtz Filter"};
constexpr auto kAdjointHelmholtzFilterName = std::string_view{"Adjoint Helmholtz Filter"};

template <typename MeshDesignVariableType, typename Function>
void for_each_design_variable(MeshDesignVariableType&& aDesignVariables,
                              const Plato::Mesh& aMesh,
                              const Function& aFunction)
{
    static_assert(std::is_same_v<std::decay_t<MeshDesignVariableType>, plato::analysis::AnalysisDomainMesh>);

    const auto& tNodeMap = aMesh->NodeMap();
    for (auto& [tBlockID, tDensityVector] : aDesignVariables.mBlockScalarField)
    {
        for (auto& tDensity : tDensityVector)
        {
            const auto tPAControlVectorIndex = tNodeMap.find(tDensity.mGlobalMeshEntityID);
            assert(tPAControlVectorIndex != tNodeMap.cend());
            aFunction(tPAControlVectorIndex->second, tDensity);
        }
    }
}

auto helmholtz_filter_parameter_list(const filter::library::FilterParameters& aFilterParameters,
                                     const std::string_view aMeshName,
                                     const std::vector<std::string>& aBlockNames,
                                     const std::string_view aPhysicsName) -> Teuchos::ParameterList
{
    Teuchos::ParameterList tParameterList;
    tParameterList.set("Physics", "Plato Driver");
    tParameterList.set("Spatial Dimension", 3);
    tParameterList.set(std::string{kInputMeshEntry}, std::string{aMeshName});
    plato_problem_sublist(tParameterList).set("Physics", std::string{aPhysicsName});
    plato_problem_sublist(tParameterList).set("PDE Constraint", std::string{kHelmholtzFilterName});
    for (const auto& tBlockName : aBlockNames)
    {
        domain_sublist(tParameterList, tBlockName).set("Element Block", tBlockName);
        domain_sublist(tParameterList, tBlockName).set("Material Model", "material_1");
    }
    // The Helmholtz filter radius needs to be scaled by 1/(2*sqrt(3)) from what
    // a typical kernel filter radius would be for it to be consistent from a user's point of view.
    constexpr double tPhysicalScaleToHelmholtzScaleFactor = 2.0 * sqrt(3);
    parameters_sublist(tParameterList)
        .set("Length Scale", aFilterParameters.mFilterRadius / tPhysicalScaleToHelmholtzScaleFactor);
    parameters_sublist(tParameterList)
        .set("Surface Length Scale", aFilterParameters.mBoundaryStickingPenalty.value_or(-1.0));
    return tParameterList;
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
    return helmholtz_filter_parameter_list(aFilterParameters, aMeshName, aBlockNames, kHelmholtzFilterName);
}

auto adjoint_helmholtz_filter_parameter_list(const filter::library::FilterParameters& aFilterParameters,
                                             const std::string_view aMeshName,
                                             const std::vector<std::string>& aBlockNames) -> Teuchos::ParameterList
{
    return helmholtz_filter_parameter_list(aFilterParameters, aMeshName, aBlockNames, kAdjointHelmholtzFilterName);
}

void update_mesh_file_name(Teuchos::ParameterList& aParameterList, const std::string_view aMeshName)
{
    aParameterList.set(std::string{kInputMeshEntry}, std::string{aMeshName});
}

Plato::ScalarVector create_control(const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh,
                                   const Plato::Mesh& aMesh)
{
    if (aAnalysisDomainMesh.mBlockScalarField.empty())
    {
        Plato::ScalarVector tControl("control", aMesh->NumNodes());
        constexpr double tFullDensity = 1.0;
        Kokkos::deep_copy(tControl, tFullDensity);
        return tControl;
    }
    else
    {
        return full_nodal_scalar_vector(aAnalysisDomainMesh, aMesh);
    }
}

std::size_t hash_current_design(const Plato::ScalarVector& aControl, const Plato::Mesh& aMesh)
{
    std::size_t tSeed = Plato::detail::hash_vector(aMesh->Coordinates());
    boost::hash_combine(tSeed, Plato::detail::hash_vector(aControl));
    return tSeed;
}

std::vector<double> scalar_vector_to_std_vector(const Plato::ScalarVector aScalarVector)
{
    const auto tHostVec = Kokkos::create_mirror_view(aScalarVector);
    Kokkos::deep_copy(tHostVec, aScalarVector);
    std::vector<double> tReturnVec;
    tReturnVec.reserve(tHostVec.size());
    std::copy(Kokkos::Experimental::cbegin(tHostVec), Kokkos::Experimental::cend(tHostVec),
              std::back_inserter(tReturnVec));
    return tReturnVec;
}

std::vector<double> design_variable_std_vector(const Plato::ScalarVector aScalarVector,
                                               const plato::analysis::AnalysisDomainMesh& aDesignVariables,
                                               const Plato::Mesh& aMesh)
{
    auto tScalarVectorOnHost = Kokkos::create_mirror_view(aScalarVector);
    Kokkos::deep_copy(tScalarVectorOnHost, aScalarVector);
    auto tDesignVariables = std::vector<double>(number_of_analysis_field_variables(aDesignVariables));

    for_each_design_variable(
        aDesignVariables, aMesh,
        [&tDesignVariables, tScalarVectorOnHost](const auto aPAControlIndex,
                                                 const plato::analysis::ScalarFieldValue& aDensity)
        { tDesignVariables[aDensity.mDesignVariableVectorIndex] = tScalarVectorOnHost[aPAControlIndex]; });
    return tDesignVariables;
}

Plato::ScalarVector full_nodal_scalar_vector(const std::vector<double>& aVector,
                                             const plato::analysis::AnalysisDomainMesh& aDesignVariables,
                                             const Plato::Mesh& aMesh,
                                             const double aFillValue)
{
    const auto tScalarVector = Plato::ScalarVector("control", aMesh->NumNodes());
    auto tScalarVectorOnHost = Kokkos::create_mirror_view(tScalarVector);

    Kokkos::deep_copy(tScalarVectorOnHost, aFillValue);

    for_each_design_variable(
        aDesignVariables, aMesh,
        [&aVector, tScalarVectorOnHost](const auto aPAControlIndex, const plato::analysis::ScalarFieldValue& aDensity)
        { tScalarVectorOnHost[aPAControlIndex] = aVector[aDensity.mDesignVariableVectorIndex]; });

    Kokkos::deep_copy(tScalarVector, tScalarVectorOnHost);
    return tScalarVector;
}

Plato::ScalarVector full_nodal_scalar_vector(const plato::analysis::AnalysisDomainMesh& aDesignVariables,
                                             const Plato::Mesh& aMesh)
{
    const auto tScalarVector = Plato::ScalarVector("control", static_cast<unsigned>(aMesh->NumNodes()));
    auto tScalarVectorOnHost = Kokkos::create_mirror_view(tScalarVector);

    constexpr auto tFixedControlValue = double{1.0};
    Kokkos::deep_copy(tScalarVectorOnHost, tFixedControlValue);

    for_each_design_variable(
        aDesignVariables, aMesh,
        [tScalarVectorOnHost](const auto aPAControlIndex, const plato::analysis::ScalarFieldValue& aDensity)
        { tScalarVectorOnHost[aPAControlIndex] = aDensity.mValue; });

    Kokkos::deep_copy(tScalarVector, tScalarVectorOnHost);
    return tScalarVector;
}

plato::analysis::AnalysisDomainMesh analysis_domain_mesh(const Plato::ScalarVector aScalarVector,
                                                         plato::analysis::AnalysisDomainMesh aDesignVariables,
                                                         const Plato::Mesh& aMesh)
{
    const auto tScalarVectorOnHost = Kokkos::create_mirror_view(aScalarVector);
    Kokkos::deep_copy(tScalarVectorOnHost, aScalarVector);

    for_each_design_variable(
        aDesignVariables, aMesh,
        [tScalarVectorOnHost](const auto aPAControlIndex, plato::analysis::ScalarFieldValue& aDensity)
        { aDensity.mValue = tScalarVectorOnHost[aPAControlIndex]; });

    return aDesignVariables;
}

std::size_t number_of_analysis_field_variables(const plato::analysis::AnalysisDomainMesh& aAnalysisDomainMesh)
{
    using Density = plato::analysis::ScalarFieldValue;
    using IndexType = plato::analysis::ScalarFieldValue::IndexType;
    using ScalarFieldVector = plato::analysis::AnalysisDomainMesh::ScalarFieldVector;

    if (aAnalysisDomainMesh.mBlockScalarField.empty())
    {
        return 0U;
    }

    const auto tLessVectorIndex = [](const Density& aDensityLeft, const Density& aDensityRight)
    { return aDensityLeft.mDesignVariableVectorIndex < aDensityRight.mDesignVariableVectorIndex; };

    const auto tMaxIndex = [&tLessVectorIndex](const ScalarFieldVector& aBlockDensities)
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
        std::transform_reduce(aAnalysisDomainMesh.mBlockScalarField.cbegin(),
                              aAnalysisDomainMesh.mBlockScalarField.cend(), IndexType{0}, tMax, tBlockDensityVector);
    return tMaxVectorIndex + 1;
}

}  // namespace plato::functional
