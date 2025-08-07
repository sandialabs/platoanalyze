#include <Kokkos_StdAlgorithms.hpp>

#include "Teuchos_UnitTestHarness.hpp"
#include "helmholtz/AdjointProblem.hpp"
#include "helmholtz/Helmholtz.hpp"
#include "util/PlatoTestHelpers.hpp"

namespace
{
auto machine() -> Plato::Comm::Machine
{
    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    return Plato::Comm::Machine{myComm};
}

using Tet4PhysicsType = ::Plato::HelmholtzFilter<Plato::Tet4>;
using Tet4ElementType = typename Tet4PhysicsType::ElementType;

auto test_mesh_params_and_machine(const int aMeshWidth)
{
    return std::make_tuple(Plato::TestHelpers::get_box_mesh("TET4", aMeshWidth),
                           Plato::TestHelpers::getParameterListForHelmholtzTest(), machine());
}

auto tet4_helmholtz_test_problem(const int aMeshWidth) -> std::shared_ptr<Plato::Helmholtz::Problem<Tet4PhysicsType>>
{
    auto [tMesh, tParameterList, tMachine] = test_mesh_params_and_machine(aMeshWidth);
    return std::make_shared<Plato::Helmholtz::Problem<Tet4PhysicsType>>(tMesh, *tParameterList, tMachine);
}

auto tet4_helmholtz_test_adjoint_problem(const int aMeshWidth)
    -> std::shared_ptr<Plato::Helmholtz::AdjointProblem<Tet4PhysicsType>>
{
    auto [tMesh, tParameterList, tMachine] = test_mesh_params_and_machine(aMeshWidth);
    return std::make_shared<Plato::Helmholtz::AdjointProblem<Tet4PhysicsType>>(tMesh, *tParameterList, tMachine);
}

auto solution_on_host(const Plato::Solutions& aSolutions, const std::string& aTag)
{
    const auto tProblemSolutionAtTag = aSolutions.get(aTag);
    const auto tSolutionMirror = Kokkos::create_mirror_view(tProblemSolutionAtTag);
    Kokkos::deep_copy(tSolutionMirror, tProblemSolutionAtTag);
    return tSolutionMirror;
}

auto test_equality_of_views(const Plato::HostScalarMultiVector aView1,
                            const Plato::HostScalarMultiVector aView2,
                            Teuchos::FancyOStream& aOut) -> bool
{
    auto tSuccess = true;
    TEUCHOS_TEST_EQUALITY(aView1.extent(0), aView2.extent(0), aOut, tSuccess);
    TEUCHOS_TEST_EQUALITY(aView1.extent(1), aView2.extent(1), aOut, tSuccess);
    constexpr auto tTolerance = 1e-14;
    if (tSuccess)
    {
        for (auto tIndex0 = unsigned{0}; tIndex0 < aView1.extent(0); ++tIndex0)
        {
            for (auto tIndex1 = unsigned{0}; tIndex1 < aView1.extent(1); ++tIndex1)
            {
                TEUCHOS_TEST_FLOATING_EQUALITY(aView1(tIndex0, tIndex1), aView2(tIndex0, tIndex1), tTolerance, aOut,
                                               tSuccess);
            }
        }
    }
    return tSuccess;
}

auto test_equality_of_solutions(const Plato::Solutions& aSolutions1,
                                const Plato::Solutions& aSolutions2,
                                Teuchos::FancyOStream& aOut) -> bool
{
    auto tSuccess = true;

    TEUCHOS_TEST_EQUALITY(aSolutions1.pde(), aSolutions2.pde(), aOut, tSuccess);
    TEUCHOS_TEST_EQUALITY(aSolutions1.physics(), aSolutions2.physics(), aOut, tSuccess);
    TEUCHOS_TEST_EQUALITY(aSolutions1.size(), aSolutions2.size(), aOut, tSuccess);
    TEUCHOS_TEST_EQUALITY(aSolutions1.tags().size(), aSolutions2.tags().size(), aOut, tSuccess);

    if (tSuccess)
    {
        for (auto tIndex = unsigned{0}; tIndex < aSolutions1.tags().size(); ++tIndex)
        {
            TEUCHOS_TEST_EQUALITY(aSolutions1.tags().at(tIndex), aSolutions2.tags().at(tIndex), aOut, tSuccess);
            const auto tSolutions1AtIndex = solution_on_host(aSolutions1, aSolutions1.tags().at(tIndex));
            const auto tSolutions2AtIndex = solution_on_host(aSolutions2, aSolutions2.tags().at(tIndex));
            TEUCHOS_TEST_EQUALITY(tSolutions1AtIndex.size(), tSolutions2AtIndex.size(), aOut, tSuccess);
            tSuccess &= test_equality_of_views(tSolutions1AtIndex, tSolutions2AtIndex, aOut);
        }
    }
    return tSuccess;
}

auto jacobian_matrix(Plato::AbstractProblem& aProblem, const std::size_t aNumberOfNodes)
{
    auto tMatrix = std::vector<std::vector<double>>{};
    tMatrix.reserve(aNumberOfNodes);
    for (auto tIndex = unsigned{0}; tIndex < aNumberOfNodes; ++tIndex)
    {
        const auto tControlOnDevice = Plato::ScalarVector{"control", aNumberOfNodes};
        const auto tControlOnHost = Kokkos::create_mirror_view(tControlOnDevice);
        tControlOnHost[tIndex] = 1.0;
        Kokkos::deep_copy(tControlOnDevice, tControlOnHost);

        const auto tJacobianRow = aProblem.criterionGradient(tControlOnDevice, Plato::Solutions{}, "dummy-name");
        Kokkos::deep_copy(tControlOnHost, tJacobianRow);

        tMatrix.emplace_back();
        tMatrix.back().reserve(aNumberOfNodes);
        std::copy(Kokkos::Experimental::begin(tControlOnHost), Kokkos::Experimental::end(tControlOnHost),
                  std::back_inserter(tMatrix.back()));
    }
    return tMatrix;
}

}  // namespace

TEUCHOS_UNIT_TEST(HelmholtzFilterTests, AdjointDirectConstruction)
{
    auto tAdjointProblemDirect = tet4_helmholtz_test_adjoint_problem(3);
    auto tProblem = tet4_helmholtz_test_problem(3);
    auto tAdjointProblemFromProblem = Plato::Helmholtz::AdjointProblem<Tet4PhysicsType>{tProblem};

    const auto tControl = Plato::ScalarVector("density", tProblem->numNodes());
    Kokkos::deep_copy(tControl, 1.0);

    const auto tSolutionFromDirect = tAdjointProblemDirect->solution(tControl);
    const auto tSolutionFromIndirect = tAdjointProblemFromProblem.solution(tControl);

    success = test_equality_of_solutions(tSolutionFromDirect, tSolutionFromIndirect, out);
}

TEUCHOS_UNIT_TEST(HelmholtzFilterTests, AdjointSizes)
{
    constexpr auto tMeshWidth = 4;
    const auto tProblem = tet4_helmholtz_test_problem(tMeshWidth);
    const auto tAdjointProblem = Plato::Helmholtz::AdjointProblem<Tet4PhysicsType>{tProblem};

    // Check sizes
    TEST_EQUALITY(tProblem->numNodes(), tAdjointProblem.numNodes());
    TEST_EQUALITY(tProblem->numCells(), tAdjointProblem.numCells());
    TEST_EQUALITY(tProblem->numDofsPerCell(), tAdjointProblem.numDofsPerCell());
    TEST_EQUALITY(tProblem->numNodesPerCell(), tAdjointProblem.numNodesPerCell());
    TEST_EQUALITY(tProblem->numDofsPerNode(), tAdjointProblem.numDofsPerNode());
    TEST_EQUALITY(tProblem->numControlsPerNode(), tAdjointProblem.numControlsPerNode());
}

TEUCHOS_UNIT_TEST(HelmholtzFilterTests, AdjointSolution)
{
    constexpr auto tMeshWidth = 4;
    const auto tProblem = tet4_helmholtz_test_problem(tMeshWidth);
    auto tAdjointProblem = Plato::Helmholtz::AdjointProblem<Tet4PhysicsType>{tProblem};

    const auto tControl = Plato::ScalarVector("density", tProblem->numNodes());
    Kokkos::deep_copy(tControl, 1.0);

    const auto tProblemSolution = tProblem->solution(tControl);
    const auto tAdjointProblemSolution = tAdjointProblem.solution(tControl);

    success = test_equality_of_solutions(tProblemSolution, tAdjointProblemSolution, out);
    success &= test_equality_of_solutions(tProblemSolution, tAdjointProblem.getSolution(), out);
}

TEUCHOS_UNIT_TEST(HelmholtzFilterTests, UnimplementedAdjointCriterionFunctions)
{
    constexpr auto tMeshWidth = 4;
    auto tAdjointProblem = Plato::Helmholtz::AdjointProblem<Tet4PhysicsType>{tet4_helmholtz_test_problem(tMeshWidth)};

    // Functions aren't implemented, just expect a throw
    const auto tDummyString = std::string{"dummy"};
    TEST_THROW(tAdjointProblem.criterionValue(Plato::ScalarVector{}, Plato::Solutions{}, tDummyString),
               std::runtime_error);
    TEST_THROW(tAdjointProblem.criterionGradientX(Plato::ScalarVector{}, Plato::Solutions{}, tDummyString),
               std::runtime_error);
}

TEUCHOS_UNIT_TEST(HelmholtzFilterTests, AdjointGradient)
{
    constexpr auto tMeshWidth = 2;
    const auto tProblem = tet4_helmholtz_test_problem(tMeshWidth);
    auto tAdjointProblem = Plato::Helmholtz::AdjointProblem<Tet4PhysicsType>{tProblem};
    const auto tNumberOfNodes = tAdjointProblem.numNodes();
    const auto tControl = Plato::ScalarVector{"control", static_cast<std::size_t>(tNumberOfNodes)};
    Kokkos::deep_copy(tControl, 1.0);

    const auto tJacobian = jacobian_matrix(*tProblem, tProblem->numNodes());
    const auto tAdjointJacobian = jacobian_matrix(tAdjointProblem, tProblem->numNodes());

    TEST_EQUALITY(tJacobian.size(), tAdjointJacobian.size());
    for (auto tRowIndex = unsigned{0}; tRowIndex < tJacobian.size(); ++tRowIndex)
    {
        TEST_EQUALITY(tJacobian.at(tRowIndex).size(), tAdjointJacobian.at(tRowIndex).size());
        for (auto tColumnIndex = unsigned{0}; tColumnIndex < tJacobian.at(tRowIndex).size(); ++tColumnIndex)
        {
            constexpr auto tTolerance = 1e-13;
            TEST_FLOATING_EQUALITY(tJacobian.at(tRowIndex).at(tColumnIndex),
                                   tAdjointJacobian.at(tColumnIndex).at(tRowIndex), tTolerance);
        }
    }
}
