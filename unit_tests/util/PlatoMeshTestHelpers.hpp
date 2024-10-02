#include <filesystem>

namespace Plato::TestHelpers {

/// @brief Writes a 3D mesh with one block and consists of a hex meshed with 6 tets at the path @a aFilePath.
void write_tet_mesh(const std::filesystem::path& aFilePath);

/// @brief Writes a 2D mesh with two blocks to disk at the path @a aFilePath.
void write_two_block_mesh(const std::filesystem::path& aFilePath);

/// @brief Writes a 2D mesh with one block to disk at the path @a aFilePath.
void write_one_block_mesh(const std::filesystem::path& aFilePath);
}  // namespace Plato::TestHelpers
