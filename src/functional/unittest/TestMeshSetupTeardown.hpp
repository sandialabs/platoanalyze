#ifndef PLATO_FUNCTIONAL_UNITTEST_TESTMESHSETUPTEARDOWN_H
#define PLATO_FUNCTIONAL_UNITTEST_TESTMESHSETUPTEARDOWN_H

#include <filesystem>
#include <plato/analysis/AnalysisDomainMesh.hpp>

#include "PlatoMesh.hpp"
#include "PlatoMeshTestHelpers.hpp"

namespace plato::functional::unittest
{
/// @brief Creates a test mesh
Plato::Mesh test_mesh(const std::filesystem::path& aMeshFilePath);

/// @brief An RAII class that writes a mesh to disk on construction and removes it on destruction.
class TestMeshSetupTeardown
{
   public:
    TestMeshSetupTeardown() = default;
    TestMeshSetupTeardown(std::string_view tTextMeshInput);
    ~TestMeshSetupTeardown();

    auto analysisDomainMeshAllDesignBlocks() const -> plato::analysis::AnalysisDomainMesh;
    auto analysisDomainMeshBlock1Fixed() const -> plato::analysis::AnalysisDomainMesh;
    auto analysisDomainMeshBlock2Fixed() const -> plato::analysis::AnalysisDomainMesh;

    auto numberOfAnalysisDomainMeshBlock1Fixed() const -> unsigned;
    auto numberOfAnalysisDomainMeshBlock2Fixed() const -> unsigned;

    auto densityValuesBlock1Fixed() const -> std::vector<double>;
    auto densityValuesBlock2Fixed() const -> std::vector<double>;

    const Plato::Mesh& mesh() const;

   private:
    std::filesystem::path mTestMeshPath = "test-mesh.exo";
    Plato::Mesh mMesh = test_mesh(mTestMeshPath);
};

}  // namespace plato::functional::unittest

#endif
