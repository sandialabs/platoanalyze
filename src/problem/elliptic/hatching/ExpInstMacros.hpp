#pragma once

#include "local_operations/optimization/NoPenalty.hpp"
#include "local_operations/optimization/Simp.hpp"

#ifdef PLATO_ALL_PENALTY
#include "local_operations/optimization/Heaviside.hpp"
#include "local_operations/optimization/Ramp.hpp"
#endif

#include "element/Bar2.hpp"
#include "element/Tet10.hpp"
#include "element/Tet4.hpp"
#include "element/Tri3.hpp"
#include "element/Tri6.hpp"

#ifdef PLATO_HEX_ELEMENTS
#include "element/Hex27.hpp"
#include "element/Hex8.hpp"
#include "element/Quad4.hpp"
#include "element/Quad9.hpp"
#endif

#include "problem/elliptic/hatching/EvaluationTypes.hpp"

#ifdef PLATO_ALL_PENALTY
#define PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T)                                       \
    template class C<Plato::Elliptic::Hatching::ResidualTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Elliptic::Hatching::ResidualTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Elliptic::Hatching::JacobianTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Elliptic::Hatching::JacobianTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Elliptic::Hatching::GradientCTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Elliptic::Hatching::GradientCTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Elliptic::Hatching::GradientXTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Elliptic::Hatching::GradientXTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Elliptic::Hatching::GradientZTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Elliptic::Hatching::GradientZTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Elliptic::Hatching::ResidualTypes<T>, Plato::RAMP>;       \
    template class C<Plato::Elliptic::Hatching::ResidualTypes<T>, Plato::Heaviside>;  \
    template class C<Plato::Elliptic::Hatching::JacobianTypes<T>, Plato::RAMP>;       \
    template class C<Plato::Elliptic::Hatching::JacobianTypes<T>, Plato::Heaviside>;  \
    template class C<Plato::Elliptic::Hatching::GradientCTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Elliptic::Hatching::GradientCTypes<T>, Plato::Heaviside>; \
    template class C<Plato::Elliptic::Hatching::GradientXTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Elliptic::Hatching::GradientXTypes<T>, Plato::Heaviside>; \
    template class C<Plato::Elliptic::Hatching::GradientZTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Elliptic::Hatching::GradientZTypes<T>, Plato::Heaviside>;
#else
#define PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T)                                       \
    template class C<Plato::Elliptic::Hatching::ResidualTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Elliptic::Hatching::ResidualTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Elliptic::Hatching::JacobianTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Elliptic::Hatching::JacobianTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Elliptic::Hatching::GradientCTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Elliptic::Hatching::GradientCTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Elliptic::Hatching::GradientXTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Elliptic::Hatching::GradientXTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Elliptic::Hatching::GradientZTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Elliptic::Hatching::GradientZTypes<T>, Plato::NoPenalty>;
#endif

#define PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T)                   \
    template class C<Plato::Elliptic::Hatching::ResidualTypes<T>>;  \
    template class C<Plato::Elliptic::Hatching::JacobianTypes<T>>;  \
    template class C<Plato::Elliptic::Hatching::GradientCTypes<T>>; \
    template class C<Plato::Elliptic::Hatching::GradientXTypes<T>>; \
    template class C<Plato::Elliptic::Hatching::GradientZTypes<T>>;

#ifdef PLATO_HEX_ELEMENTS
#define PLATO_ELLIPTIC_HATCHING_EXP_INST(C, T)             \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Tet4>);  \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Tri3>);  \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Tet10>); \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Hex8>);  \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Quad4>); \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Hex27>);

#define PLATO_ELLIPTIC_HATCHING_EXP_INST_2(C, T)             \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Tet4>);  \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Tri3>);  \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Tet10>); \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Hex8>);  \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Quad4>); \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Hex27>);
#else
#define PLATO_ELLIPTIC_HATCHING_EXP_INST(C, T)            \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Tet4>); \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Tri3>); \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_(C, T<Plato::Tet10>);

#define PLATO_ELLIPTIC_HATCHING_EXP_INST_2(C, T)            \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Tet4>); \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Tri3>); \
    PLATO_ELLIPTIC_HATCHING_EXP_INST_2_(C, T<Plato::Tet10>);
#endif
