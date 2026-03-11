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
#include "problem/hyperbolic/EvaluationTypes.hpp"

#define PLATO_HYPERBOLIC_EXP_INST_(C, T)                                      \
    template class C<Plato::Hyperbolic::ResidualTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Hyperbolic::ResidualTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Hyperbolic::GradientUTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Hyperbolic::GradientUTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Hyperbolic::GradientVTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Hyperbolic::GradientVTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Hyperbolic::GradientATypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Hyperbolic::GradientATypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Hyperbolic::GradientXTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Hyperbolic::GradientXTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Hyperbolic::GradientZTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Hyperbolic::GradientZTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Hyperbolic::ResidualTypes<T>, Plato::RAMP>;       \
    template class C<Plato::Hyperbolic::ResidualTypes<T>, Plato::Heaviside>;  \
    template class C<Plato::Hyperbolic::GradientUTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Hyperbolic::GradientUTypes<T>, Plato::Heaviside>; \
    template class C<Plato::Hyperbolic::GradientVTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Hyperbolic::GradientVTypes<T>, Plato::Heaviside>; \
    template class C<Plato::Hyperbolic::GradientATypes<T>, Plato::RAMP>;      \
    template class C<Plato::Hyperbolic::GradientATypes<T>, Plato::Heaviside>; \
    template class C<Plato::Hyperbolic::GradientXTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Hyperbolic::GradientXTypes<T>, Plato::Heaviside>; \
    template class C<Plato::Hyperbolic::GradientZTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Hyperbolic::GradientZTypes<T>, Plato::Heaviside>;

#define PLATO_HYPERBOLIC_EXP_INST_2_(C, T)                  \
    template class C<Plato::Hyperbolic::ResidualTypes<T>>;  \
    template class C<Plato::Hyperbolic::GradientUTypes<T>>; \
    template class C<Plato::Hyperbolic::GradientVTypes<T>>; \
    template class C<Plato::Hyperbolic::GradientATypes<T>>; \
    template class C<Plato::Hyperbolic::GradientXTypes<T>>; \
    template class C<Plato::Hyperbolic::GradientZTypes<T>>;

#define PLATO_HYPERBOLIC_EXP_INST(C, T)             \
    PLATO_HYPERBOLIC_EXP_INST_(C, T<Plato::Tet4>);  \
    PLATO_HYPERBOLIC_EXP_INST_(C, T<Plato::Tri3>);  \
    PLATO_HYPERBOLIC_EXP_INST_(C, T<Plato::Tet10>); \
    PLATO_HYPERBOLIC_EXP_INST_(C, T<Plato::Hex8>);  \
    PLATO_HYPERBOLIC_EXP_INST_(C, T<Plato::Quad4>); \
    PLATO_HYPERBOLIC_EXP_INST_(C, T<Plato::Hex27>);

#define PLATO_HYPERBOLIC_EXP_INST_2(C, T)             \
    PLATO_HYPERBOLIC_EXP_INST_2_(C, T<Plato::Tet4>);  \
    PLATO_HYPERBOLIC_EXP_INST_2_(C, T<Plato::Tri3>);  \
    PLATO_HYPERBOLIC_EXP_INST_2_(C, T<Plato::Tet10>); \
    PLATO_HYPERBOLIC_EXP_INST_2_(C, T<Plato::Hex8>);  \
    PLATO_HYPERBOLIC_EXP_INST_2_(C, T<Plato::Quad4>); \
    PLATO_HYPERBOLIC_EXP_INST_2_(C, T<Plato::Hex27>);
