#include "TestMeshSetupTeardown.hpp"

namespace plato::functional::unittest
{
namespace
{
// A vector of nodal densities with densities equal to the global ID. This can be used for tests that include all blocks
// (none fixed).
const auto tDensityVector1AllBlocks =
    std::vector<plato::mesh::Density>{{1, 0, 1.0}, {2, 1, 2.0}, {3, 2, 3.0}, {4, 3, 4.0}, {5, 4, 5.0}};
const auto tDensityVector2AllBlocks =
    std::vector<plato::mesh::Density>{{3, 2, 3.0}, {4, 3, 4.0}, {5, 4, 5.0}, {6, 5, 6.0}, {7, 6, 7.0}, {8, 7, 8.0}};

// A vector of nodal densities for block 1 with densities equal to the vector index assuming block 2 is fixed.
// The densities are set to the vector index.
const auto tDensityVector1Block2Fixed =
    std::vector<plato::mesh::Density>{{1, 0, 1.0}, {2, 1, 2.0}, {3, 2, 3.0}, {4, 3, 4.0}, {5, 4, 5.0}};

// A vector of nodal densities for block 2 with densities equal to the vector index assuming block 1 is fixed.
// The densities are set to the vector index.
const auto tDensityVector2Block1Fixed =
    std::vector<plato::mesh::Density>{{3, 0, 3.0}, {4, 1, 4.0}, {5, 2, 5.0}, {6, 3, 6.0}, {7, 4, 7.0}, {8, 5, 8.0}};

std::vector<double> density_values(const std::vector<plato::mesh::Density>& aDensityVector)
{
    auto tDensityValues = std::vector<double>{};
    tDensityValues.reserve(aDensityVector.size());
    std::transform(aDensityVector.cbegin(), aDensityVector.cend(), std::back_inserter(tDensityValues),
                   [](const auto& tDensity) { return tDensity.mDensity; });
    return tDensityValues;
}

}  // namespace

Plato::Mesh test_mesh(const std::filesystem::path& aMeshFilePath)
{
    Plato::TestHelpers::write_two_block_mesh(aMeshFilePath);
    return Plato::MeshFactory::create(aMeshFilePath.string());
}

TestMeshSetupTeardown::~TestMeshSetupTeardown() { std::filesystem::remove(mTestMeshPath); }

auto TestMeshSetupTeardown::meshDesignVariablesAllDesignBlocks() const -> plato::mesh::MeshDesignVariables
{
    return plato::mesh::MeshDesignVariables{mTestMeshPath,
                                            {{1, tDensityVector1AllBlocks}, {2, tDensityVector2AllBlocks}}};
}

auto TestMeshSetupTeardown::meshDesignVariablesBlock1Fixed() const -> plato::mesh::MeshDesignVariables
{
    return plato::mesh::MeshDesignVariables{mTestMeshPath, {{2, tDensityVector2Block1Fixed}}};
}

auto TestMeshSetupTeardown::meshDesignVariablesBlock2Fixed() const -> plato::mesh::MeshDesignVariables
{
    return plato::mesh::MeshDesignVariables{mTestMeshPath, {{1, tDensityVector1Block2Fixed}}};
}

auto TestMeshSetupTeardown::numberOfMeshDesignVariablesBlock1Fixed() const -> unsigned
{
    return tDensityVector2Block1Fixed.size();
}

auto TestMeshSetupTeardown::numberOfMeshDesignVariablesBlock2Fixed() const -> unsigned
{
    return tDensityVector1Block2Fixed.size();
}

auto TestMeshSetupTeardown::densityValuesBlock1Fixed() const -> std::vector<double>
{
    return density_values(tDensityVector2Block1Fixed);
}

auto TestMeshSetupTeardown::densityValuesBlock2Fixed() const -> std::vector<double>
{
    return density_values(tDensityVector1Block2Fixed);
}

const Plato::Mesh& TestMeshSetupTeardown::mesh() const { return mMesh; }

}  // namespace plato::functional::unittest
