#include "PlatoMeshTestHelpers.hpp"

#include <mpi.h>

#include <stk_io/FillMesh.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/BulkData.hpp>
#include <stk_mesh/base/Field.hpp>
#include <stk_mesh/base/MeshBuilder.hpp>
#include <string_view>

namespace Plato::TestHelpers {

namespace
{
void write_text_mesh(const std::string_view aMeshDescription, const std::filesystem::path& aFilePath)
{
  auto tBulk = stk::mesh::MeshBuilder(MPI_COMM_SELF).create();
  tBulk->mesh_meta_data().use_simple_fields();
  stk::io::fill_mesh(std::string{aMeshDescription}, *tBulk);

  stk::io::StkMeshIoBroker tIOBroker;
  tIOBroker.set_bulk_data(std::move(tBulk));
  const size_t outputFileIndex = tIOBroker.create_output_mesh(aFilePath.string(), stk::io::WRITE_RESULTS);
  tIOBroker.write_output_mesh(outputFileIndex);
  tIOBroker.write_defined_output_fields(outputFileIndex);
}
}

void write_two_block_mesh(const std::filesystem::path& aFilePath) {
  constexpr auto tTwoDTriMesh = std::string_view{
      "textmesh:"
      "0,1,TRI_3_2D,3,1,4,block_1\n"
      "0,2,TRI_3_2D,1,2,4,block_1\n"
      "0,3,TRI_3_2D,2,5,4,block_1\n"
      "0,4,TRI_3_2D,4,5,7,block_2\n"
      "0,5,TRI_3_2D,7,8,4,block_2\n"
      "0,6,TRI_3_2D,3,4,8,block_2\n"
      "0,7,TRI_3_2D,8,6,3,block_2\n"
      "|coordinates: 0,0,0.125,0,0,0.125,0.0625,0.125,0.125,0.125,0,0.25,0.125,0.25,0.0625,0.25"
      "|dimension:2"};
  write_text_mesh(tTwoDTriMesh, aFilePath);
}

void write_one_block_mesh(const std::filesystem::path& aFilePath) {
  constexpr auto tTwoDTriMesh = std::string_view{
      "textmesh:"
      "0,1,TRI_3_2D,1,2,3,block_1\n"
      "0,2,TRI_3_2D,2,4,3,block_1\n"
      "|coordinates: 0,0, 1,0, 0,1, 1,1"
      "|dimension:2"};

  write_text_mesh(tTwoDTriMesh, aFilePath);
}

}  // namespace Plato::TestHelpers
