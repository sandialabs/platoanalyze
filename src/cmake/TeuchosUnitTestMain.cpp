#include <mpi.h>

#include <Kokkos_Core.hpp>

#include "Teuchos_UnitTestRepository.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

#ifdef WATCH_ARITHMETIC
#include <fenv.h>
#endif

int main(int argc, char* argv[])
{
    auto tThreadsProvided = int{};
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &tThreadsProvided);
    assert(tThreadsProvided == MPI_THREAD_FUNNELED);

    Kokkos::initialize(argc, argv);
    Plato::MeshFactory::initialize(argc, argv);

#ifdef WATCH_ARITHMETIC
    feclearexcept(FE_ALL_EXCEPT);
    feenableexcept(FE_ALL_EXCEPT - FE_INEXACT - FE_UNDERFLOW);
#endif

    auto result = Teuchos::UnitTestRepository::runUnitTestsFromMain(argc, argv);

    Plato::MeshFactory::finalize();
    Kokkos::finalize();
    MPI_Finalize();
    std::cout << "RESULT: " << result << std::endl;
    return result;
}
