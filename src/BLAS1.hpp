/*
 * BLAS1.hpp
 *
 *  Created on: Apr 17, 2020
 */

#pragma once

#include <Kokkos_Macros.hpp>
#include <KokkosBlas1_fill.hpp>
#include <KokkosBlas1_scal.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace blas1
{

/******************************************************************************//**
 * \fn device_type inline void dot
 *
 * \brief Compute absolute value of a one-dimensional scalar array
 *
 * \param [in/out] aVector 1D scalar view
**********************************************************************************/
inline void abs(const Plato::ScalarVector & aVector)
{
    Plato::OrdinalType tLength = aVector.size();
    Kokkos::parallel_for("calculate absolute value", Kokkos::RangePolicy<>(0, tLength), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    {
        aVector(aOrdinal) = fabs(aVector(aOrdinal));
    });
}
// function abs

/******************************************************************************//**
 * \brief Fill host 1D container with random numbers
 * \param [in] aLowerBound lower bounds on random numbers
 * \param [in] aUpperBound upper bounds on random numbers
 * \param [in] aOutput output 1D container
**********************************************************************************/
template<typename VecType>
inline void random(const Plato::Scalar & aLowerBound, const Plato::Scalar & aUpperBound, VecType & aOutput)
{
    unsigned int tRANDOM_SEED = 1;
    std::srand(tRANDOM_SEED);
    const Plato::OrdinalType tSize = aOutput.size();
    for(Plato::OrdinalType tIndex = 0; tIndex < tSize; tIndex++)
    {
        const Plato::Scalar tRandNum = static_cast<Plato::Scalar>(std::rand()) / static_cast<Plato::Scalar>(RAND_MAX);
        aOutput(tIndex) = aLowerBound + ( (aUpperBound - aLowerBound) * tRandNum);
    }
}
// function random

/******************************************************************************//**
 * \brief Compute inner product
 * \param [in] aVec1 1D Kokkos container
 * \param [in] aVec2 1D Kokkos container
 * \return inner product
**********************************************************************************/
template<typename VecOneT, typename VecTwoT>
inline Plato::Scalar dot(const VecOneT & aVec1, const VecTwoT & aVec2)
{
    if(aVec2.size() != aVec1.size())
    {
        std::stringstream tMsg;
        tMsg << "BLAS1 DOT: DIMENSION MISMATCH. VECTOR ONE HAS SIZE = " << aVec1.size() << " AND VECTOR TWO HAS SIZE = "
            << aVec2.size() << ".VECTOR ONE HAS LABEL '" << aVec1.label() << "' AND VECTOR TWO HAS LABEL '"
            << aVec2.label() << "'.\n";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    Plato::Scalar tOutput = 0.;
    const Plato::OrdinalType tSize = aVec1.size();
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tSize),
                            KOKKOS_LAMBDA(const Plato::OrdinalType & aIndex, Plato::Scalar & aSum)
    {
        aSum += aVec1(aIndex) * aVec2(aIndex);
    }, tOutput);
    return (tOutput);
}
// function dot

/******************************************************************************//**
 * \brief Compute Euclidean norm of a vector
 * \param [in] aVector 1D Kokkos container
 * \return norm/length
**********************************************************************************/
template<typename VecOneT>
inline Plato::Scalar norm(const VecOneT & aVector)
{
    if(aVector.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::stringstream tMsg;
        tMsg << "BLAS 1 NORM: INPUT VECTOR WITH LABEL '" << aVector.label() << "' IS EMPTY.\n";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }
    const auto tDot = Plato::blas1::dot(aVector, aVector);
    const auto tOutput = std::sqrt(tDot);
    return (tOutput);
}
// function norm

/******************************************************************************//**
 * \brief Set all the elements to a scalar value
 * \param [in] aInput   scalar value
 * \param [out] aVector 1D Kokkos container
**********************************************************************************/
template<typename VectorT>
inline void fill(const Plato::Scalar & aInput, const VectorT & aVector)
{
    if(aVector.size() <= static_cast<Plato::OrdinalType>(0))
    {
	return;
    }

    if(std::isfinite(aInput) == false)
    {
        ANALYZE_THROWERR("BLAS 1 FILL: INPUT SCALAR IS NOT A FINITE NUMBER.\n")
    }
    KokkosBlas::fill(aVector, aInput);
}
// function fill

/******************************************************************************//**
 * \brief Set all the elements to a scalar value
 * \param [in] aMultiplier scalar multiplier
 * \param [in] aOrdinals   list of entry ordinals
 * \param [in] aValues     list of scalar values (1D Kokkos container)
 * \param [out] aVector    1D Kokkos container
**********************************************************************************/
template<typename VectorT>
inline void fill(const Plato::Scalar & aMultiplier,
                 const Plato::OrdinalVector & aOrdinals,
                 const Plato::ScalarVector & aValues,
                 const VectorT & aOutput)
{
    if(aOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::stringstream tMsg;
        tMsg << "BLAS 1 FILL: OUTPUT VECTOR WITH LABEL '" << aOutput.label() << "' IS EMPTY.\n";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    if(aValues.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::stringstream tMsg;
        tMsg << "BLAS 1 FILL: INPUT VECTOR WITH LABEL '" << aValues.label() << "' IS EMPTY.\n";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    if(std::isfinite(aMultiplier) == false)
    {
        ANALYZE_THROWERR("BLAS 1 FILL: SCALAR MULTIPLIER IS NOT A FINITE NUMBER.")
    }

    if(aOrdinals.size() != aValues.size())
    {
        std::ostringstream tMsg;
        tMsg << "BLAS 1 FILL: DIMENSION MISMATCH. INPUT LIST OF ORDINALS AND VALUES HAVE DIFFERENT LENGTH. "
                << "LIST OF ORDINALS WITH LABEL '" << aOrdinals.label() << "' HAS LENGTH '"<< aOrdinals.size()
                << "' AND LIST OF VALUES WITH LABEL '" << aValues.label() << "' HAS LENGTH '" << aValues.size() << "'\n.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    const Plato::OrdinalType tNumLocalVals = aOutput.size();
    Kokkos::parallel_for("fill vector", Kokkos::RangePolicy<>(0, tNumLocalVals), KOKKOS_LAMBDA(const Plato::OrdinalType & aIndex)
    {
        aOutput(aOrdinals(aIndex)) = aMultiplier * aValues(aIndex);
    });
}
// function fill

/******************************************************************************//**
 * \brief Copy input 1D container into output 1D container
 * \param [in] aInput   1D Kokkos container
 * \param [out] aOutput 1D Kokkos container
**********************************************************************************/
template<typename VecOneT, typename VecTwoT>
inline void copy(const VecOneT & aInput, const VecTwoT & aOutput)
{
    if(aInput.size() != aOutput.size())
    {
        std::stringstream tMsg;
        tMsg << "BLAS 1 COPY: DIMENSION MISMATCH. INPUT VECTOR WITH LABEL '" << aInput.label() << "' HAS LENGTH '"
        << aInput.size() << "' AND OUTPUT VECTOR WITH LABEL '" << aOutput.label() << "' HAS LENGTH '" << aOutput.size() << "'.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for("copy vector", Kokkos::RangePolicy<>(0, tNumLocalVals), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) = aInput(aOrdinal);
    });
}
// function copy

/******************************************************************************//**
 * \brief Scale all the elements by input scalar value
 * \param [in] aInput   scalar value
 * \param [out] aOutput 1D Kokkos container
**********************************************************************************/
template<typename VecT>
inline void scale(const Plato::Scalar & aInput, const VecT & aVector)
{
    if(std::isfinite(aInput) == false)
    {
        ANALYZE_THROWERR("BLAS 1 SCALE: SCALAR MULTIPLIER IS NOT A FINITE NUMBER.")
    }

    if(aVector.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::stringstream tMsg;
        tMsg << "BLAS 1 SCALE: INPUT VECTOR WITH LABEL '" << aVector.label() << "' IS EMPTY.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }
    KokkosBlas::scal(aVector, aInput, aVector);
}
// function scale

/******************************************************************************//**
 * \brief Update elements of B with scaled values of A, /f$ B = B + alpha*A /f$
 * \param [in] aAlpha   multiplier of 1D container A
 * \param [in] aInput   input 1D Kokkos container
 * \param [out] aOutput output 1D Kokkos container
**********************************************************************************/
template<typename VecT>
inline void axpy(const Plato::Scalar & aAlpha, const VecT & aInput, const VecT & aOutput)
{
    if(aInput.size() != aOutput.size())
    {
        std::stringstream tMsg;
        tMsg << "DIMENSION MISMATCH. INPUT VECTOR HAS SIZE = " << aInput.size()
                << " AND OUTPUT VECTOR HAS SIZE = " << aOutput.size() << ".\n";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for("Plato::axpy", Kokkos::RangePolicy<>(0, tNumLocalVals), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) += aAlpha * aInput(aOrdinal);
    });
}
// function axpy

/******************************************************************************//**
 * \brief Update elements of B with scaled values of A, /f$ B = beta*B + alpha*A /f$
 * \param [in] aAlpha multiplier of 1D container A
 * \param [in] aInput   input 1D Kokkos container
 * \param [in] aBeta    multiplier of 1D container B
 * \param [out] aOutput output 1D Kokkos container
**********************************************************************************/
template<typename VecT>
void update(const Plato::Scalar & aAlpha, const VecT & aInput, const Plato::Scalar & aBeta, const VecT & aOutput)
{
    if(std::isfinite(aAlpha) == false)
    {
        ANALYZE_THROWERR("BLAS 1 UPDATE: SCALAR MULTIPLIER 'ALPHA' IS NOT A FINITE NUMBER.")
    }

    if(std::isfinite(aBeta) == false)
    {
        ANALYZE_THROWERR("BLAS 1 UPDATE: SCALAR MULTIPLIER 'BETA' IS NOT A FINITE NUMBER.")
    }

    if(aInput.size() != aOutput.size())
    {
        std::stringstream tMsg;
        tMsg << "BLAS 1 UPDATE: DIMENSION MISMATCH. INPUT VECTOR WITH LABEL '" << aInput.label() << "' HAS LENGTH '"
                << aInput.size() << "' AND OUTPUT VECTOR WITH LABEL '" << aOutput.label() << "' HAS LENGTH '"
                << aOutput.size() << "'.\n";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    Plato::OrdinalType tNumLocalVals = aInput.size();
    Kokkos::parallel_for("update vector", Kokkos::RangePolicy<>(0, tNumLocalVals), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aOrdinal) = aAlpha * aInput(aOrdinal) + aBeta * aOutput(aOrdinal);
    });
}
// function update

/******************************************************************************//**
 * \brief Reduced operation: sum all the elements in input array and return local sum
 * \param [in] aInput 1D Kokkos container
 * \param [out] aOutput local sum
**********************************************************************************/
template<typename VecT, typename ScalarT>
void local_sum(const VecT & aInput, ScalarT & aOutput)
{
    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::stringstream tMsg;
        tMsg << "BLAS 1 LOCAL_SUM: INPUT VECTOR WITH LABEL '" << aInput.label() << "' IS EMPTY.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    ScalarT tOutput = 0.0;
    const Plato::OrdinalType tNumLocalElems = aInput.size();
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumLocalElems), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal, ScalarT & aLocalSum)
    {
      aLocalSum += aInput(aCellOrdinal);
    }, tOutput);
    aOutput = tOutput;
}
// function local_sum

/******************************************************************************//**
 * \brief Compute the global maximum element in range.
 * \param [in] aInput array of elements
 * \param [out] aOutput maximum element
**********************************************************************************/
template<typename VecT, typename ScalarT>
void max(const VecT & aInput, ScalarT & aOutput)
{
    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::ostringstream tMsg;
        tMsg << "BLAS 1 MAX: INPUT VECTOR WITH LABEL '" << aInput.label() << "' IS EMPTY.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    const Plato::OrdinalType tSize = aInput.size();
    //const ScalarT* tInputData = aInput.data();
    aOutput = 0.0;

    Kokkos::Max<ScalarT> tMaxReducer(aOutput);
    Kokkos::parallel_reduce("KokkosReductionOperations::max",
                            Kokkos::RangePolicy<>(0, tSize),
                            KOKKOS_LAMBDA(const OrdinalType & aIndex, ScalarT & aValue){
        tMaxReducer.join(aValue, aInput[aIndex]);
    }, tMaxReducer);
}
// function max

/******************************************************************************//**
 * \brief Compute the global minimum element in range.
 * \param [in] aInput array of elements
 * \param [out] aOutput minimum element
**********************************************************************************/
template<typename VecT, typename ScalarT>
void min(const VecT & aInput, ScalarT & aOutput)
{
    if(aInput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::ostringstream tMsg;
        tMsg << "BLAS 1 MIN: INPUT VECTOR WITH LABEL '" << aInput.label() << "' IS EMPTY.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    const Plato::OrdinalType tSize = aInput.size();
    //const ScalarT* tInputData = aInput.data();
    aOutput = 0.0;

    Kokkos::Min<ScalarT> tMinReducer(aOutput);
    Kokkos::parallel_reduce("KokkosReductionOperations::min",
                            Kokkos::RangePolicy<>(0, tSize),
                            KOKKOS_LAMBDA(const OrdinalType & aIndex, ScalarT & aValue){
        tMinReducer.join(aValue, aInput[aIndex]);
    }, tMinReducer);
}
// function min

/******************************************************************************//**
 * \brief Extract a sub array
 *
 * \tparam NumStride stride, e.g. number of degree of freedom per node
 * \tparam NumOffset offset, e.g. degree of freedom offset
 *
 * \param [in] aFromVector input array
 * \param [out] aToVector  extracted sub-array
 *
 * aToVector(i) = aFromVector(i*NumStride+NumOffset)
 *
**********************************************************************************/
template<int NumStride, int NumOffset>
inline void extract(const Plato::ScalarVector& aFromVector, Plato::ScalarVector& aToVector)
{
    if(aFromVector.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::ostringstream tMsg;
        tMsg << "BLAS 1 EXTRACT: FROM VECTOR WITH LABEL '" << aFromVector.label() << "' IS EMPTY.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    if(aToVector.size() <= static_cast<Plato::OrdinalType>(0))
    {
        std::ostringstream tMsg;
        tMsg << "BLAS 1 EXTRACT: TO VECTOR WITH LABEL '" << aToVector.label() << "' IS EMPTY.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    auto tNumRows = aToVector.extent(0);

    Kokkos::parallel_for("extract", Kokkos::RangePolicy<>(0, tNumRows), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    {
        aToVector(aOrdinal) = aFromVector(aOrdinal*NumStride + NumOffset);
    });
}
// function extract

/******************************************************************************//**
 * \fn inline Plato::Scalar inf_norm
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Calculate misfit between two fields and return Inf norm of misfit/residual vector.
 *
 * \param [in] aNumNodes number of nodes in the mesh
 * \param [in] aFieldOne physical field one
 * \param [in] aFieldTwo physical field two
 *
 * \return euclidean norm scalar
 *
 **********************************************************************************/
inline Plato::Scalar inf_norm
(const Plato::ScalarVector& aFieldOne,
 const Plato::ScalarVector& aFieldTwo)
{
    Plato::ScalarVector tResidual("residual", aFieldOne.size());
    Plato::blas1::copy(aFieldTwo, tResidual);
    Plato::blas1::update(1.0, aFieldOne, -1.0, tResidual);

    Plato::Scalar tOutput = 0.0;
    Plato::blas1::abs(tResidual);
    Plato::blas1::max(tResidual, tOutput);

    return tOutput;
}
// function inf_norm

/******************************************************************************//**
 * \fn inline Plato::Scalar norm
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Calculate misfit between two fields and return Euclidean norm of misfit/residual vector.
 *
 * \param [in] aNumNodes number of nodes in the mesh
 * \param [in] aFieldOne physical field one
 * \param [in] aFieldTwo physical field two
 *
 * \return Euclidean norm scalar
 *
 **********************************************************************************/
inline Plato::Scalar norm
(const Plato::ScalarVector& aFieldOne,
 const Plato::ScalarVector& aFieldTwo)
{
    Plato::ScalarVector tResidual("residual", aFieldOne.size());
    Plato::blas1::copy(aFieldTwo, tResidual);
    Plato::blas1::update(1.0, aFieldOne, -1.0, tResidual);
    auto tValue = Plato::blas1::norm(tResidual);
    return tValue;
}
// function norm

}
// namespace blas1

}
// namespace Plato
