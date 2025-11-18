#ifndef PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_FINITEDEFORMATIONMECHANICS_H
#define PLATO_ELLIPTIC_FINITEDEFORMATIONMECHANICS_FINITEDEFORMATIONMECHANICS_H

#include "element/MechanicsElement.hpp"
#include "problem/elliptic/finite_deformation_mechanics/Factory.hpp"

namespace plato::elliptic::finite_deformation_mechanics
{
/// @brief struct specifying element and function factory types for finite deformation mechanics physics.
/// @tparam TopoElementType is the base topological element (e.g. tri3, tet4, hex8, etc.) upon which the element type
/// defining degrees of freedom per node, etc. for finite deformation mechanics relies.
template <typename TopoElementType>
struct FiniteDeformationMechanics
{
    using ElementType = Plato::MechanicsElement<TopoElementType>;
    using FunctionFactory = FiniteDeformationMechanicsFactory;
};
}  // namespace plato::elliptic::finite_deformation_mechanics

#endif
