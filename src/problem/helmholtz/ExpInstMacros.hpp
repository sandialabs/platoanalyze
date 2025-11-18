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
#include "problem/helmholtz/EvaluationTypes.hpp"

#define SKIP_HELMHOLTZ_EXP_INST

#ifdef SKIP_HELMHOLTZ_EXP_INST
#define PLATO_HELMHOLTZ_DEF_3_(C, T)
#define PLATO_HELMHOLTZ_DEC_3_(C, T)
#define PLATO_HELMHOLTZ_DEF_3(C, T)
#define PLATO_HELMHOLTZ_DEC_3(C, T)
#else

#define PLATO_HELMHOLTZ_DEF_3_(C, T)                             \
    extern template class C<Plato::Helmholtz::ResidualTypes<T>>; \
    extern template class C<Plato::Helmholtz::JacobianTypes<T>>;

#define PLATO_HELMHOLTZ_DEC_3_(C, T)                      \
    template class C<Plato::Helmholtz::ResidualTypes<T>>; \
    template class C<Plato::Helmholtz::JacobianTypes<T>>;

#define PLATO_HELMHOLTZ_DEF_3(C, T)             \
    PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Tet4>);  \
    PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Tri3>);  \
    PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Tet10>); \
    PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Hex8>);  \
    PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Quad4>); \
    PLATO_HELMHOLTZ_DEF_3_(C, T<Plato::Hex27>);

#define PLATO_HELMHOLTZ_DEC_3(C, T)             \
    PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Tet4>);  \
    PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Tri3>);  \
    PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Tet10>); \
    PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Hex8>);  \
    PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Quad4>); \
    PLATO_HELMHOLTZ_DEC_3_(C, T<Plato::Hex27>);

#endif
