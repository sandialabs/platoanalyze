#pragma once

#include "PlatoStaticsTypes.hpp"
#include "PlatoMathTypes.hpp"

namespace Plato::Hyperbolic::Micromorphic
{

/***************************************************************************//**
 * \brief Functor for projecting full stress tensor (not just symmetric) to nodes. 
*******************************************************************************/
template<typename ElementType,
         Plato::OrdinalType DofOffset = 0>
class ProjectStressToNode : ElementType
{
private:
    using ElementType::mNumSpatialDims;
    using ElementType::mNumNodesPerCell; 
    using ElementType::mNumDofsPerNode;  

    using ElementType::mNumVoigtTerms;  
    using ElementType::mNumFullTerms;  

    // 2-D Example: mVoigtMap[0] = 0, mVoigtMap[1] = 1, mVoigtMap[2] = 2, mVoigtMap[3] = 2,
    // where the stress tensor in Voigt notation is given s = {s_11, s_22, s_12} (plane strain)
    Plato::OrdinalType mVoigtMap[mNumFullTerms]; /*!< matrix with indices to stress tensor entries in Voigt notation */
    // 2-D Example: mSkewScale[0] = 0, mSkewScale[1] = 0, mSkewScale[2] = 1, mSkewScale[3] = -1,
    // where the skew stress tensor in Voigt storage is given s = {0, 0, s_12} (plane strain)
    Plato::Scalar mSkewScale[mNumFullTerms]; /*!< matrix with indices to scalr factors for skew terms with Voigt storage */

public:
    ProjectStressToNode()
    {
        this->initializeVoigtMap();
        this->initializeSkewScale();
    }

    // overloaded for cauchy and micro stresses
    template<typename ProjectedScalarType, typename StressScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void 
    operator()
    (      Plato::OrdinalType                               aCellOrdinal,
           Plato::OrdinalType                               aGpOrdinal,
           Plato::ScalarMultiVectorT<ProjectedScalarType>   aOutput,
     const Plato::ScalarArray3DT<StressScalarType>        & aSymmetricMesoStress,
     const Plato::ScalarArray3DT<StressScalarType>        & aSkewMesoStress,
     const Plato::ScalarArray3DT<StressScalarType>        & aSymmetricMicroStress,
     const Plato::Array<mNumNodesPerCell, Plato::Scalar>  & aBasisFunctions,
     const Plato::ScalarMultiVectorT<VolumeScalarType>    & aVolume) const
    {
        this->addSymmetricStressAtNodes(aCellOrdinal,aGpOrdinal,aOutput,aSymmetricMicroStress,aBasisFunctions,aVolume,1.0);
        this->addSymmetricStressAtNodes(aCellOrdinal,aGpOrdinal,aOutput,aSymmetricMesoStress,aBasisFunctions,aVolume,-1.0);
        this->addSkewStressAtNodes(aCellOrdinal,aGpOrdinal,aOutput,aSkewMesoStress,aBasisFunctions,aVolume,-1.0);
    }

    // overloaded for inertia stresses
    template<typename ProjectedScalarType, typename StressScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void 
    operator()
    (      Plato::OrdinalType                               aCellOrdinal,
           Plato::OrdinalType                               aGpOrdinal,
           Plato::ScalarMultiVectorT<ProjectedScalarType>   aOutput,
     const Plato::ScalarArray3DT<StressScalarType>        & aSymmetricMicroStress,
     const Plato::ScalarArray3DT<StressScalarType>        & aSkewMicroStress,
     const Plato::Array<mNumNodesPerCell, Plato::Scalar>  & aBasisFunctions,
     const Plato::ScalarMultiVectorT<VolumeScalarType>    & aVolume) const
    {
        this->addSymmetricStressAtNodes(aCellOrdinal,aGpOrdinal,aOutput,aSymmetricMicroStress,aBasisFunctions,aVolume,1.0);
        this->addSkewStressAtNodes(aCellOrdinal,aGpOrdinal,aOutput,aSkewMicroStress,aBasisFunctions,aVolume,1.0);
    }

private:

    inline void 
    initializeVoigtMap()
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumVoigtTerms; tDofIndex++)
        {
            mVoigtMap[tDofIndex] = tVoigtTerm++;
        }
        tVoigtTerm = mNumSpatialDims;
        for(Plato::OrdinalType tDofIndex = mNumVoigtTerms; tDofIndex < mNumFullTerms; tDofIndex++)
        {
            mVoigtMap[tDofIndex] = tVoigtTerm++;
        }
    }

    inline void 
    initializeSkewScale()
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumSpatialDims; tDofIndex++)
        {
            mSkewScale[tDofIndex] = 0.0;
        }
        for(Plato::OrdinalType tDofIndex = mNumSpatialDims; tDofIndex < mNumVoigtTerms; tDofIndex++)
        {
            mSkewScale[tDofIndex] = 1.0;
        }
        for(Plato::OrdinalType tDofIndex = mNumVoigtTerms; tDofIndex < mNumFullTerms; tDofIndex++)
        {
            mSkewScale[tDofIndex] = -1.0;
        }
    }

    template<typename ProjectedScalarType, typename StressScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void 
    addSymmetricStressAtNodes
    (      Plato::OrdinalType                               aCellOrdinal,
           Plato::OrdinalType                               aGpOrdinal,
           Plato::ScalarMultiVectorT<ProjectedScalarType>   aOutput,
     const Plato::ScalarArray3DT<StressScalarType>        & aStress,
     const Plato::Array<mNumNodesPerCell, Plato::Scalar>  & aBasisFunctions,
     const Plato::ScalarMultiVectorT<VolumeScalarType>    & aVolume,
           Plato::Scalar                                    aScale) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumFullTerms; tDofIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex + DofOffset;
                ProjectedScalarType tResult =
                    aScale * aVolume(aCellOrdinal,aGpOrdinal) * 
                    aStress(aCellOrdinal,aGpOrdinal,mVoigtMap[tDofIndex]) * 
                    aBasisFunctions(tNodeIndex);
                Kokkos::atomic_add(&aOutput(aCellOrdinal,tLocalOrdinal), tResult);
            }
        }
    }

    template<typename ProjectedScalarType, typename StressScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void 
    addSkewStressAtNodes
    (      Plato::OrdinalType                               aCellOrdinal,
           Plato::OrdinalType                               aGpOrdinal,
           Plato::ScalarMultiVectorT<ProjectedScalarType>   aOutput,
     const Plato::ScalarArray3DT<StressScalarType>        & aStress,
     const Plato::Array<mNumNodesPerCell, Plato::Scalar>  & aBasisFunctions,
     const Plato::ScalarMultiVectorT<VolumeScalarType>    & aVolume,
           Plato::Scalar                                    aScale) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumFullTerms; tDofIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex + DofOffset;
                ProjectedScalarType tResult = 
                    aScale * mSkewScale[tDofIndex] * aVolume(aCellOrdinal,aGpOrdinal) * 
                    aStress(aCellOrdinal,aGpOrdinal,mVoigtMap[tDofIndex]) * 
                    aBasisFunctions(tNodeIndex);
                Kokkos::atomic_add(&aOutput(aCellOrdinal,tLocalOrdinal), tResult);
            }
        }
    }

};

} 
