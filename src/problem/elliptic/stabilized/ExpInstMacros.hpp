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

#include "problem/elliptic/stabilized/EvaluationTypes.hpp"

#ifdef PLATO_ALL_PENALTY
#define PLATO_STABILIZED_EXP_INST_(C, T)                                      \
    template class C<Plato::Stabilized::ResidualTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Stabilized::ResidualTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Stabilized::JacobianTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Stabilized::JacobianTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Stabilized::GradientXTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Stabilized::GradientXTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Stabilized::GradientZTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Stabilized::GradientZTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Stabilized::ResidualTypes<T>, Plato::RAMP>;       \
    template class C<Plato::Stabilized::ResidualTypes<T>, Plato::Heaviside>;  \
    template class C<Plato::Stabilized::JacobianTypes<T>, Plato::RAMP>;       \
    template class C<Plato::Stabilized::JacobianTypes<T>, Plato::Heaviside>;  \
    template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::Heaviside>; \
    template class C<Plato::Stabilized::GradientXTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Stabilized::GradientXTypes<T>, Plato::Heaviside>; \
    template class C<Plato::Stabilized::GradientZTypes<T>, Plato::RAMP>;      \
    template class C<Plato::Stabilized::GradientZTypes<T>, Plato::Heaviside>;
#else
#define PLATO_STABILIZED_EXP_INST_(C, T)                                      \
    template class C<Plato::Stabilized::ResidualTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Stabilized::ResidualTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Stabilized::JacobianTypes<T>, Plato::MSIMP>;      \
    template class C<Plato::Stabilized::JacobianTypes<T>, Plato::NoPenalty>;  \
    template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Stabilized::JacobianNTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Stabilized::GradientXTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Stabilized::GradientXTypes<T>, Plato::NoPenalty>; \
    template class C<Plato::Stabilized::GradientZTypes<T>, Plato::MSIMP>;     \
    template class C<Plato::Stabilized::GradientZTypes<T>, Plato::NoPenalty>;
#endif

#ifdef PLATO_HEX_ELEMENTS
#define PLATO_STABILIZED_EXP_INST(C, T)             \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tet4>);  \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tri3>);  \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tet10>); \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Hex8>);  \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Quad4>); \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Hex27>);
#else
#define PLATO_STABILIZED_EXP_INST(C, T)            \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tet4>); \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tri3>); \
    PLATO_STABILIZED_EXP_INST_(C, T<Plato::Tet10>);
#endif

#ifdef PLATO_ALL_PENALTY
#define PLATO_STABILIZED_EXP_INST_2_(C, T, P, D)                                                               \
    template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::RAMP>;                                                                             \
    template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::Heaviside>;                                                                        \
    template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::RAMP>;                                                                             \
    template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::Heaviside>;                                                                        \
    template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::RAMP>;                                                                             \
    template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::Heaviside>;                                                                        \
    template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::RAMP>;                                                                             \
    template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::Heaviside>;                                                                        \
    template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::RAMP>;                                                                             \
    template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::Heaviside>;
#else
#define PLATO_STABILIZED_EXP_INST_2_(C, T, P, D)                                                               \
    template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::ResidualTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::JacobianTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>,  \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::JacobianNTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::GradientXTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::NoPenalty>;                                                                        \
    template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::MSIMP>;                                                                            \
    template class C<Plato::Stabilized::GradientZTypes<T<D, P<D>::mNumDofsPerNode, P<D>::mPressureDofOffset>>, \
                     Plato::NoPenalty>;
#endif

#ifdef PLATO_HEX_ELEMENTS
#define PLATO_STABILIZED_EXP_INST_2(C, T, P)             \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tet4);  \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tri3);  \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tet10); \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Hex8);  \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Quad4); \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Hex27);
#else
#define PLATO_STABILIZED_EXP_INST_2(C, T, P)            \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tet4); \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tri3); \
    PLATO_STABILIZED_EXP_INST_2_(C, T, P, Plato::Tet10);
#endif
