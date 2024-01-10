/*
 * SurfacePressureIntegral.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "SpatialModel.hpp"
#include "NaturalBCData.hpp"
#include "WeightedNormalVector.hpp"

namespace Plato
{
/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type uniform or variable pressure.
 *
 * \tparam SpatialDim   spatial dimension
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<
  typename ElementType,
  Plato::OrdinalType NumDofs=ElementType::mNumSpatialDims,
  Plato::OrdinalType DofsPerNode=NumDofs,
  Plato::OrdinalType DofOffset=0 >
class SurfacePressureIntegral final
{
private:
    const std::string mSidesetName;
    const Plato::Scalar mCurrentTime;
    std::unique_ptr<NaturalBCData<NumDofs>> mBCData;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \pre @a aBCData must not be null. Checked with an assertion.
     **********************************************************************************/
    SurfacePressureIntegral(const std::string aSidesetName,
                            const Plato::Scalar aCurrentTime,
                            std::unique_ptr<NaturalBCData<NumDofs>> aBCData);

    /***************************************************************************//**
     * \brief Evaluate natural boundary condition surface integrals.
     *
     * \tparam StateScalarType   state forward automatically differentiated (FAD) type
     * \tparam ControlScalarType control FAD type
     * \tparam ConfigScalarType  configuration FAD type
     * \tparam ResultScalarType  result FAD type
     *
     * \param [in]  aSpatialModel Plato Analyze spatial model.
     * \param [in]  aState        2-D view of state variables.
     * \param [in]  aControl      2-D view of control variables.
     * \param [in]  aConfig       3-D view of configuration variables.
     * \param [out] aResult       Assembled vector to which the boundary terms will be added
     * \param [in]  aScale        scalar multiplier
     *
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void operator()(
        const Plato::SpatialModel                          & aSpatialModel,
        const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
        const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
              Plato::Scalar aScale) const;
};
// class SurfacePressureIntegral

/***************************************************************************//**
 * \brief SurfacePressureIntegral::SurfacePressureIntegral constructor definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfacePressureIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::SurfacePressureIntegral
(const std::string aSidesetName,
 const Plato::Scalar aCurrentTime,
 std::unique_ptr<NaturalBCData<NumDofs>> aBCData) :
    mSidesetName(aSidesetName),
    mCurrentTime(aCurrentTime),
    mBCData(std::move(aBCData))
{
    assert(mBCData);
}
// class SurfacePressureIntegral::SurfacePressureIntegral

/***************************************************************************//**
 * \brief SurfacePressureIntegral::operator() function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfacePressureIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::operator()(
  const Plato::SpatialModel                          & aSpatialModel,
  const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
  const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
  const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
        Plato::Scalar aScale
) const
{
    const auto tElementOrds  = aSpatialModel.Mesh->GetSideSetElements(mSidesetName);
    const auto tNodeOrds     = aSpatialModel.Mesh->GetSideSetLocalNodes(mSidesetName);
    const auto tConnectivity = aSpatialModel.Mesh->Connectivity();

    const Plato::OrdinalType tNumFaces = tElementOrds.size();

    const Plato::WeightedNormalVector<ElementType> weightedNormalVector;

    const auto tBoundaryData = mBCData->getScalarData(aSpatialModel.Mesh, mCurrentTime);

    const auto tCubatureWeights = ElementType::Face::getCubWeights();
    const auto tCubaturePoints  = ElementType::Face::getCubPoints();
    const auto tNumPoints = tCubatureWeights.size();

    // pressure forces should act towards the surface; thus, -1.0 is used to invert the outward facing normal inwards.
    constexpr Plato::Scalar tNormalMultiplier = -1.0;
    Kokkos::parallel_for("surface pressure integral", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
        const auto tElementOrdinal = tElementOrds(aSideOrdinal);

        Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
        for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
        {
            tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
        }

        const auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
        const auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
        const auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
        const auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

        // compute area weighted normal vector
        Plato::Array<ElementType::mNumSpatialDims, ConfigScalarType> tWeightedNormalVec;
        weightedNormalVector(tElementOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tWeightedNormalVec);

        // project into aResult workset
        for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
        {
            const auto tGlobalNodeOrdinal = tConnectivity(tElementOrdinal*ElementType::mNumNodesPerCell + tLocalNodeOrds(tNode));
            const Plato::Scalar tPressure = scalarBoundaryDataAtIndex(tBoundaryData, tGlobalNodeOrdinal);
            for( Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
            {
                const auto tElementDofOrdinal = (tLocalNodeOrds[tNode] * DofsPerNode) + tDof + DofOffset;
                const ResultScalarType tVal = 
                  tWeightedNormalVec(tDof) * tPressure * aScale * tCubatureWeight * tNormalMultiplier * tBasisValues(tNode);
                Kokkos::atomic_add(&aResult(tElementOrdinal, tElementDofOrdinal), tVal);
            }
        }
    });
}
// class SurfacePressureIntegral::operator()

}
// namespace Plato
