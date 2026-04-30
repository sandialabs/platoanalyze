#include "problem/parabolic/test_utilities/DataUtilities.hpp"

#include <vector>

#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "test_utilities/PlatoTestHelpers.hpp"

namespace plato::parabolic::test_utilities
{
auto multi_dimension_view_from_vector(const std::vector<std::vector<double>>& aStatesVector) -> Plato::ScalarMultiVector
{
    const Plato::OrdinalType tNumSteps = aStatesVector.size();
    assert(tNumSteps > 0);
    const Plato::OrdinalType tNumDofs = aStatesVector[0].size();
    Plato::ScalarMultiVector tMultiVector("", tNumSteps, tNumDofs);
    for (Plato::OrdinalType tStep = 0; tStep < tNumSteps; tStep++)
    {
        const auto tStateView = Plato::TestHelpers::create_device_view(aStatesVector[tStep]);
        Kokkos::parallel_for(
            "multidimensional view", Kokkos::RangePolicy<int>(0, tNumDofs),
            KOKKOS_LAMBDA(Plato::OrdinalType tDofOrdinal) {
                tMultiVector(tStep, tDofOrdinal) = tStateView(tDofOrdinal);
            });
    }
    return tMultiVector;
}
}  // namespace plato::parabolic::test_utilities
