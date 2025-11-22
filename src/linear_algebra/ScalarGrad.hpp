#pragma once

#include "linear_algebra/PlatoMathTypes.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! \brief Scalar gradient functor.
 *
 *  Given a gradient matrix and scalar field, compute the scalar gradient.
 *
 ******************************************************************************/
template <typename ElementType>
class ScalarGrad
{
   public:
    /***********************************************************************************
     * \brief Compute scalar field gradient
     * \param [in] aCellOrdinal cell ordinal
     * \param [in/out] aOutput scalar field gradient workset
     * \param [in] aScalarField scalar field workset
     * \param [in] aGradient configuration gradient workset
     **********************************************************************************/
    template <typename OutputScalarType, typename StateScalarType, typename ConfigScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::Array<ElementType::mNumSpatialDims, OutputScalarType>& aOutput,
        Plato::ScalarMultiVectorT<StateScalarType> aScalarField,
        const Plato::Matrix<ElementType::mNumNodesPerCell, ElementType::mNumSpatialDims, ConfigScalarType>& aGradient)
        const
    {
        // compute scalar gradient
        //
        for (Plato::OrdinalType tDimIndex = 0; tDimIndex < ElementType::mNumSpatialDims; tDimIndex++)
        {
            aOutput(tDimIndex) = 0.0;
            for (Plato::OrdinalType tNodeIndex = 0; tNodeIndex < ElementType::mNumNodesPerCell; tNodeIndex++)
            {
                aOutput(tDimIndex) += aScalarField(aCellOrdinal, tNodeIndex) * aGradient(tNodeIndex, tDimIndex);
            }
        }
    }
};
// class ScalarGrad

}  // namespace Plato
// namespace Plato
