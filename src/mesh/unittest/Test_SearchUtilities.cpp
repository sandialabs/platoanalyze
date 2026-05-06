#include <Teuchos_UnitTestHarness.hpp>
#include <string>

#include "core_types/PlatoTypes.hpp"
#include "domain/SpatialModel.hpp"
#include "element/Tet4.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"
#include "mesh/SearchUtilities.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::mesh::unittest
{
namespace
{
const std::string kTet4MeshType{"TET4"};

Teuchos::ParameterList box_mesh_spatial_model_param_list()
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");
    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Element Block", "body");
    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Material Model", "Vaseline");
    return tParameterList;
}

Plato::ScalarMultiVector get_node_set_coordinates(const Plato::Mesh aMesh,
                                                  const Plato::OrdinalVectorT<const Plato::OrdinalType>& aNodes)
{
    Plato::OrdinalType tNumNodes = aNodes.size();
    Plato::OrdinalType tSpaceDim = aMesh->NumDimensions();
    Plato::ScalarMultiVector tLocations("node locations", tSpaceDim, tNumNodes);

    auto tCoords = aMesh->Coordinates();
    Kokkos::parallel_for(
        "get coords", Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes), KOKKOS_LAMBDA(int nodeOrdinal) {
            auto tNodeOrdinal = aNodes(nodeOrdinal);
            for (Plato::OrdinalType iDim = 0; iDim < tSpaceDim; iDim++)
                tLocations(iDim, nodeOrdinal) = tCoords(tNodeOrdinal * tSpaceDim + iDim);
        });

    return tLocations;
}
}  // namespace

TEUCHOS_UNIT_TEST(SearchUtilities, FindParentElements_SameDomainDifferentNodesets)
{
    constexpr Plato::OrdinalType tMeshWidth = 1;
    const auto tMesh = Plato::TestHelpers::get_box_mesh(kTet4MeshType, tMeshWidth);

    Plato::DataMap tMap{};
    Teuchos::ParameterList tParamList = box_mesh_spatial_model_param_list();
    const auto tParsedDomains = plato::domain::parse_domains(tParamList, tMesh);
    plato::domain::SpatialModel tSpatialModel(tMesh, tParsedDomains, tMap);

    const auto tChildNodes = tMesh->GetNodeSetNodes("x-");
    const auto tParentNodes = tMesh->GetNodeSetNodes("x+");
    auto tChildLocations = get_node_set_coordinates(tMesh, tChildNodes);
    auto tParentLocations = get_node_set_coordinates(tMesh, tParentNodes);

    assert(tSpatialModel.mDomains.size() == 1);
    const auto tOnlyDomain = tSpatialModel.mDomains.front();

    const Plato::OrdinalVector tParentElements("parent elements", tChildNodes.size());
    plato::mesh::find_parent_elements<Plato::Tet4, Plato::Scalar>(tSpatialModel.mMesh, tOnlyDomain.cellOrdinals(),
                                                                  tChildLocations, tParentLocations, tParentElements);

    const auto tParentElementsHost = Plato::TestHelpers::get(tParentElements);
    const std::vector<Plato::OrdinalType> tGoldParents{4, 3, 0, 0};
    for (Plato::OrdinalType tIndex = 0; tIndex < tParentElementsHost.size(); tIndex++)
    {
        TEST_EQUALITY(tParentElementsHost[tIndex], tGoldParents[tIndex]);
    }
}
}  // namespace plato::mesh::unittest
