/*
 * SurfaceLoadIntegral.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include "FadTypes.hpp"
#include "NaturalBCData.hpp"
#include "SpatialModel.hpp"
#include "SurfaceArea.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: UNIFORM and UNIFORM COMPONENT.
 *
 * \tparam ElementType  Element type (e.g., MechanicsElement<Tet10>)
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<
  typename ElementType,
  Plato::OrdinalType NumDofs=ElementType::mNumSpatialDims,
  Plato::OrdinalType DofsPerNode=NumDofs,
  Plato::OrdinalType DofOffset=0 >
class SurfaceLoadIntegral final
{
private:
    const std::string mSideSetName; /*!< side set name */
    const Plato::Scalar mCurrentTime;
    std::unique_ptr<NaturalBCData<NumDofs>> mBCData;

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    SurfaceLoadIntegral(const std::string & aSideSetName, 
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
     * \param [in]  aSpatialModel Plato spatial model
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
// class SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::SurfaceLoadIntegral constructor definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfaceLoadIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::SurfaceLoadIntegral
(const std::string & aSideSetName, 
const Plato::Scalar aCurrentTime,
std::unique_ptr<NaturalBCData<NumDofs>> aBCData) :
    mSideSetName(aSideSetName),
    mCurrentTime(aCurrentTime),
    mBCData(std::move(aBCData))
{
    assert(mBCData);
}
// class SurfaceLoadIntegral::SurfaceLoadIntegral

/***************************************************************************//**
 * \brief SurfaceLoadIntegral::operator() function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfaceLoadIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::operator()(
    const Plato::SpatialModel                          & aSpatialModel,
    const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
    const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
          Plato::Scalar aScale
) const
{
    const auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    const auto tNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    const auto tConnectivity = aSpatialModel.Mesh->Connectivity();

    const Plato::OrdinalType tNumFaces = tElementOrds.size();

    const Plato::SurfaceArea<ElementType> surfaceArea;

    const auto tBoundaryData = mBCData->getVectorData(aSpatialModel.Mesh, mCurrentTime);
    const auto tCubatureWeights = ElementType::Face::getCubWeights();
    const auto tCubaturePoints  = ElementType::Face::getCubPoints();
    const auto tNumPoints = tCubatureWeights.size();

    Kokkos::parallel_for("surface load integral", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
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

      ResultScalarType tSurfaceArea(0.0);
      surfaceArea(tElementOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tSurfaceArea);
      tSurfaceArea *= aScale;
      tSurfaceArea *= tCubatureWeight;

      // project into aResult workset
      for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
      {
          const auto tGlobalNodeOrdinal = tConnectivity(tElementOrdinal*ElementType::mNumNodesPerCell + tLocalNodeOrds(tNode));
          const auto tFlux = vectorBoundaryDataAtIndex<NumDofs>(tBoundaryData, tGlobalNodeOrdinal);
          for( Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
          {
              const auto tElementDofOrdinal = tLocalNodeOrds[tNode] * DofsPerNode + tDof + DofOffset;
              const ResultScalarType tResult = tBasisValues(tNode)*tFlux(tDof)*tSurfaceArea;
              Kokkos::atomic_add(&aResult(tElementOrdinal,tElementDofOrdinal), tResult);
          }
      }
    });
}
// class SurfaceLoadIntegral::operator()

}
// namespace Plato
