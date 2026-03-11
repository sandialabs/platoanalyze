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
#include "problem/geometric/EvaluationTypes.hpp"

#define PLATO_GEOMETRIC_EXP_INST_2_(C, T)                  \
    template class C<Plato::Geometric::ResidualTypes<T>>;  \
    template class C<Plato::Geometric::GradientXTypes<T>>; \
    template class C<Plato::Geometric::GradientZTypes<T>>;

#define PLATO_GEOMETRIC_EXP_INST(C, T)             \
    PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tet4>);  \
    PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tri3>);  \
    PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Tet10>); \
    PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Hex8>);  \
    PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Quad4>); \
    PLATO_GEOMETRIC_EXP_INST_(C, T<Plato::Hex27>);

#define PLATO_GEOMETRIC_EXP_INST_2(C, T)             \
    PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tet4>);  \
    PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tri3>);  \
    PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Tet10>); \
    PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Hex8>);  \
    PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Quad4>); \
    PLATO_GEOMETRIC_EXP_INST_2_(C, T<Plato::Hex27>);
