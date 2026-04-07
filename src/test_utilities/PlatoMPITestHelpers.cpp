#include "test_utilities/PlatoMPITestHelpers.hpp"

namespace Plato::TestHelpers
{
auto duplicate_comm_world() -> Plato::Comm::Machine
{
    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    return Plato::Comm::Machine(myComm);
}
}  // namespace Plato::TestHelpers
