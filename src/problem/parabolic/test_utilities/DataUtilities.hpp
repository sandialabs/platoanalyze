#ifndef PLATO_PROBLEM_PARABOLIC_TESTUTILITIES_DATAUTILITIES
#define PLATO_PROBLEM_PARABOLIC_TESTUTILITIES_DATAUTILITIES

#include <vector>

#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace plato::parabolic::test_utilities
{
[[nodiscard]] auto multi_dimension_view_from_vector(const std::vector<std::vector<double>>& aStatesVector)
    -> Plato::ScalarMultiVector;
}

#endif
