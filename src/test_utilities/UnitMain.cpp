#include <mpi.h>

#include <Kokkos_Core.hpp>
#include <Teuchos_UnitTestRepository.hpp>

#include "PlatoMesh.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    Plato::MeshFactory::initialize(argc, argv);

    auto result = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

    Plato::MeshFactory::finalize();
    Kokkos::finalize();
    MPI_Finalize();

    return result;
}
