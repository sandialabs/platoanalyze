#pragma once

#include "element/Bar2.hpp"
#include "element/Hex27.hpp"
#include "element/Hex8.hpp"
#include "element/Quad4.hpp"
#include "element/Quad9.hpp"
#include "element/Tet10.hpp"
#include "element/Tet4.hpp"
#include "element/Tri3.hpp"
#include "element/Tri6.hpp"
#include "local_operations/optimization/Heaviside.hpp"
#include "local_operations/optimization/NoPenalty.hpp"
#include "local_operations/optimization/Ramp.hpp"
#include "local_operations/optimization/Simp.hpp"
#include "problem/elliptic/EvaluationTypes.hpp"

#define PLATO_ELLIPTIC_EXP_INST_(C, T)                                      \
    template class C<Plato::Elliptic::ResidualTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Elliptic::ResidualTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Elliptic::ResidualTypes<T>, Plato::RAMP>;       \
    template class C<Plato::Elliptic::ResidualTypes<T>, Plato::Heaviside>;  \
    template class C<Plato::Elliptic::JacobianTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Elliptic::JacobianTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Elliptic::JacobianTypes<T>, Plato::RAMP>;       \
    template class C<Plato::Elliptic::JacobianTypes<T>, Plato::Heaviside>;  \
    template class C<Plato::Elliptic::GradientXTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Elliptic::GradientXTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Elliptic::GradientXTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Elliptic::GradientXTypes<T>, Plato::Heaviside>; \
    template class C<Plato::Elliptic::GradientZTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Elliptic::GradientZTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Elliptic::GradientZTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Elliptic::GradientZTypes<T>, Plato::Heaviside>;

#define PLATO_ELLIPTIC_EXP_INST_2_(C, T)                  \
    template class C<Plato::Elliptic::ResidualTypes<T>>;  \
    template class C<Plato::Elliptic::JacobianTypes<T>>;  \
    template class C<Plato::Elliptic::GradientXTypes<T>>; \
    template class C<Plato::Elliptic::GradientZTypes<T>>;

#define PLATO_ELLIPTIC_EXP_INST(C, T)             \
    PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tet4>);  \
    PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tri3>);  \
    PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Tet10>); \
    PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Hex8>);  \
    PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Quad4>); \
    PLATO_ELLIPTIC_EXP_INST_(C, T<Plato::Hex27>);

#define PLATO_ELLIPTIC_EXP_INST_2(C, T)             \
    PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tet4>);  \
    PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tri3>);  \
    PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Tet10>); \
    PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Hex8>);  \
    PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Quad4>); \
    PLATO_ELLIPTIC_EXP_INST_2_(C, T<Plato::Hex27>);
