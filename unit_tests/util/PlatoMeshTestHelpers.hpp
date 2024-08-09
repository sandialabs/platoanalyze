#include <filesystem>

namespace Plato::TestHelpers {

/// @brief Writes a 2D mesh with two blocks to disk at the path @a aFilePath.
void write_two_block_mesh(const std::filesystem::path& aFilePath);

/// @brief Writes a 2D mesh with a non-trivial nodemap to the path @a aFilePath
void write_mesh_with_nodemap(const std::filesystem::path& aFilePath);
}  // namespace Plato::TestHelpers
