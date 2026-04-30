#ifndef PLATO_TEST_UTILITIES_PLATOMPITESTHELPERS
#define PLATO_TEST_UTILITIES_PLATOMPITESTHELPERS

#include "utilities/ParallelComm.hpp"

namespace Plato::TestHelpers
{
/// @brief constructs a Plato::Comm::Machine from MPI_COMM_WORLD.
[[nodiscard]] auto duplicate_comm_world() -> Plato::Comm::Machine;
}  // namespace Plato::TestHelpers

#endif
