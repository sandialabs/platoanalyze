#pragma once

#include <Sacado.hpp>

#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace Plato
{

template <typename TopoElementType>
struct Config
{
    using FadType =
        Sacado::Fad::SFad<Plato::Scalar, TopoElementType::mNumSpatialDims * TopoElementType::mNumNodesPerCell>;
};

template <typename ElementType>
struct FadTypes
{
    using StateFad = Sacado::Fad::SFad<Plato::Scalar, ElementType::mNumDofsPerNode * ElementType::mNumNodesPerCell>;
    using LocalStateFad = Sacado::Fad::SFad<Plato::Scalar, ElementType::mNumLocalDofsPerCell>;
    using ControlFad = Sacado::Fad::SFad<Plato::Scalar, ElementType::mNumNodesPerCell>;
    using ConfigFad = typename Plato::Config<typename ElementType::TopoElementType>::FadType;
    using NodeStateFad =
        Sacado::Fad::SFad<Plato::Scalar, ElementType::mNumNodeStatePerNode * ElementType::mNumNodesPerCell>;
};

// is_fad<TypesT, T>::value is true if T is of any AD type defined TypesT.
//
template <typename TypesT, typename T>
struct is_fad
{
    static constexpr bool value =
        std::is_same<T, typename TypesT::StateFad>::value || std::is_same<T, typename TypesT::ControlFad>::value ||
        std::is_same<T, typename TypesT::ConfigFad>::value || std::is_same<T, typename TypesT::NodeStateFad>::value ||
        std::is_same<T, typename TypesT::LocalStateFad>::value;
};

// which_fad<TypesT,T1,T2>::type returns:
// -- compile error  if T1 and T2 are both AD types defined in TypesT,
// -- T1             if only T1 is an AD type in TypesT,
// -- T2             if only T2 is an AD type in TypesT,
// -- T2             if neither are AD types.
//
template <typename TypesT, typename T1, typename T2>
struct which_fad
{
    static_assert(!(is_fad<TypesT, T1>::value && is_fad<TypesT, T2>::value),
                  "Only one template argument can be an AD type.");
    using type = typename std::conditional<is_fad<TypesT, T1>::value, T1, T2>::type;
};

// fad_type_t<PhysicsT,T1,T2,T3,...,TN> returns:
// -- compile error  if more than one of T1,...,TN is an AD type in SimplexFadTypes<PhysicsT>,
// -- type TI        if only TI is AD type in SimplexFadTypes<PhysicsT>,
// -- TN             if none of TI are AD type in SimplexFadTypes<PhysicsT>.
//
template <typename TypesT, typename... P>
struct fad_type;
template <typename TypesT, typename T>
struct fad_type<TypesT, T>
{
    using type = T;
};
template <typename TypesT, typename T, typename... P>
struct fad_type<TypesT, T, P...>
{
    using type = typename which_fad<TypesT, T, typename fad_type<TypesT, P...>::type>::type;
};
template <typename PhysicsT, typename... P>
using fad_type_t = typename fad_type<FadTypes<PhysicsT>, P...>::type;

}  // namespace Plato
