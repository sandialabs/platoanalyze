#pragma once
#include <iostream>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <string>
#include <math.h> 

#include <Sacado.hpp>
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Evaluator
{

namespace Math
{

template <typename T>
T copy(const T& aArray)
{
  T tArray;
  tArray.mData = typename T::data_type("data", aArray.mData.extent(0));
  Kokkos::deep_copy(tArray.mData, aArray.mData);
  return tArray;
}

} // end namespace Math


template <typename Real = double, typename Int = int>
class RealArray
{
  public:

  using data_type = Plato::ScalarVectorT<Real>;
  using array_type = RealArray<Real,Int>;

  data_type mData;

  // Constructors
  RealArray()=default;
  RealArray(array_type &&)=delete;
  RealArray(array_type const &)=default;
  array_type& operator=(array_type&&)=default;
  array_type& operator=(const array_type&)=default;

  explicit RealArray(Int aLength, Real aInit=0.0)
  {
    mData = data_type("data", aLength);
    Kokkos::deep_copy(mData, aInit);
  }

  RealArray(data_type const & aData)
  {
    mData = aData;
  }

  // Unary operators
  array_type operator- () {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    Kokkos::parallel_for("operator-", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) = -tA(aOrdinal);
    });
    return tArray;
  }

  array_type operator+ () {
   return Math::copy(*this);
  }

  // binary operators 
  array_type operator+(const array_type& b) {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator+", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) += tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator-(const array_type& b) {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator-", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) -= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator*(const array_type& b) {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator*", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) *= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator/(const array_type& b) {
    auto tArray = Math::copy(*this);
    auto tA = tArray.mData;
    auto tB = b.mData;
    Kokkos::parallel_for("operator/", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
    {
      tA(aOrdinal) /= tB(aOrdinal);
    });
    return tArray;
  }

  array_type operator=(const Real& b) {
    Kokkos::deep_copy(mData, b);
    return *this;
  }

  // I/O
  template <typename fReal, typename fInt>
  friend
  std::ostream& operator<<(std::ostream& os, const RealArray<fReal,fInt>& aArray);

};

namespace Math
{

template <typename Real, typename Int>
RealArray<Real,Int> pow(const RealArray<Real,Int>& aArray, double aExp)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("pow", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::pow(tB(aOrdinal), aExp);
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> pow(const RealArray<Real,Int>& aArray, const RealArray<Real,Int>& aExp)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  auto tC = aExp.mData;
  Kokkos::parallel_for("pow", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::pow(tB(aOrdinal), tC(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> sin(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("sin", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::sin(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> cos(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("cos", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::cos(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> tan(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("tan", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::tan(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> asin(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("asin", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::asin(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> acos(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("acos", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::acos(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> atan(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("atan", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::atan(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> sinh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("sinh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::sinh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> cosh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("cosh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::cosh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> tanh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("tanh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::tanh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> asinh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("asinh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::asinh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> acosh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("acosh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::acosh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> atanh(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("atanh", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::atanh(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> log(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("log", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::log(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> log10(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("log10", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::log10(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> exp(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("exp", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::exp(tB(aOrdinal));
  });
  return tArray;
}

template <typename Real, typename Int>
RealArray<Real,Int> sqrt(const RealArray<Real,Int>& aArray)
{
  auto tArray = copy(aArray);
  auto tA = tArray.mData;
  auto tB = aArray.mData;
  Kokkos::parallel_for("sqrt", Kokkos::RangePolicy<Int>(0,tArray.mData.extent(0)), KOKKOS_LAMBDA(Int aOrdinal)
  {
    tA(aOrdinal) = std::sqrt(tB(aOrdinal));
  });
  return tArray;
}

}

template <typename Real = double, typename Int = int>
std::ostream& operator<<(std::ostream& os, const RealArray<Real,Int>& aArray)
{
    auto tData_Host = Kokkos::create_mirror_view(aArray.mData);
    Kokkos::deep_copy(tData_Host, aArray.mData);
    for (const Real& v : tData_Host)
    {
      os << " " << v << " ";
    }
    return os;
}

} // end namespace Evaluator

} // end namespace Plato
