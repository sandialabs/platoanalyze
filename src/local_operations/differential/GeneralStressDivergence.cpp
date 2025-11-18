#include "local_operations/differential/GeneralStressDivergence.hpp"

#include "linear_algebra/PlatoMathTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace plato::composable_function_objects::shape_function_operations
{
namespace
{
const Plato::Matrix<2, 2, Plato::OrdinalType> kVoigt2DMap{0, 2, 2, 1};

const Plato::Matrix<3, 3, Plato::OrdinalType> kVoigt3DMap{0, 5, 4, 5, 1, 3, 4, 3, 2};
}  // namespace

namespace detail
{
Plato::Matrix<2, 2, Plato::OrdinalType> voigt_map_2d() { return kVoigt2DMap; }

Plato::Matrix<3, 3, Plato::OrdinalType> voigt_map_3d() { return kVoigt3DMap; }
}  // namespace detail
}  // namespace plato::composable_function_objects::shape_function_operations
