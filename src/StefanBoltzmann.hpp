#pragma once

#include "FadTypes.hpp"
#include "SpatialModel.hpp"
#include "SurfaceArea.hpp"
#include "ExpressionEvaluator.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: STEFAN_BOLTZMANN
 *
 * \tparam ElementType  Element type (e.g., MechanicsElement<Tet10>)
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<
  typename ElementType,
  Plato::OrdinalType DofsPerNode,
  Plato::OrdinalType DofOffset=0 >
class StefanBoltzmann
{
private:
    const std::string mSideSetName; /*!< side set name */

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    StefanBoltzmann(
        const std::string & aSideSetName);

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

    template<typename StateScalarType>
    void evaluateSurfaceFlux(
      Plato::ScalarMultiVectorT<StateScalarType> tFlux,
      Plato::ScalarMultiVectorT<StateScalarType> tState,
      Plato::OrdinalVectorT<const Plato::OrdinalType> tSideSetElementOrds,
      Plato::OrdinalVectorT<const Plato::OrdinalType> tSideSetLocalNodeOrds) const;

}; // class StefanBoltzmann

/***************************************************************************//**
 * \brief StefanBoltzmann::StefanBoltzmann constructor definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
StefanBoltzmann<ElementType, DofsPerNode, DofOffset>::StefanBoltzmann(
  const std::string & aSideSetName
) :
    mSideSetName(aSideSetName)
{
}

template<typename ElementType, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType>
void StefanBoltzmann<ElementType, DofsPerNode, DofOffset>::evaluateSurfaceFlux(
      Plato::ScalarMultiVectorT<StateScalarType> aFlux,
      Plato::ScalarMultiVectorT<StateScalarType> aState,
      Plato::OrdinalVectorT<const Plato::OrdinalType> aSideSetElementOrds,
      Plato::OrdinalVectorT<const Plato::OrdinalType> aSideSetLocalNodeOrds
) const
{
    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tCubaturePoints  = ElementType::Face::getCubPoints();
    auto tNumPoints = tCubatureWeights.size();
    Plato::OrdinalType tNumFaces = aSideSetElementOrds.size();

    Kokkos::parallel_for("compute flux", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumFaces, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType iGpOrdinal)
    {
        auto tElementOrdinal = aSideSetElementOrds(aSideOrdinal);
        auto tCubPoint = tCubaturePoints(iGpOrdinal);
        auto tBasisValues = ElementType::Face::basisValues(tCubPoint);
    
        StateScalarType tState(0.0);
        for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
        {
            auto tLocalNodeOrd = aSideSetLocalNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNode);
            tState += tBasisValues(tNode)*aState(tElementOrdinal, DofsPerNode*tLocalNodeOrd+DofOffset);
        }
        if(tState > 0.0)
        {
          StateScalarType tStateToFourth = pow(tState,4);
          aFlux(aSideOrdinal, iGpOrdinal) = -tStateToFourth * 5.67e-8; // TODO
        }
    });
}


/***************************************************************************//**
 * \brief StefanBoltzmann::operator() function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void StefanBoltzmann<ElementType, DofsPerNode, DofOffset>::operator()(
    const Plato::SpatialModel                          & aSpatialModel,
    const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
    const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
          Plato::Scalar aScale
) const
{
    auto tElementOrds = aSpatialModel.Mesh->GetSideSetElements(mSideSetName);
    auto tNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(mSideSetName);
    Plato::OrdinalType tNumFaces = tElementOrds.size();

    Plato::SurfaceArea<ElementType> surfaceArea;

    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tCubaturePoints  = ElementType::Face::getCubPoints();
    auto tNumPoints = tCubatureWeights.size();

    Plato::ScalarMultiVectorT<StateScalarType> tFlux("fluxes", tNumFaces, tNumPoints);
    evaluateSurfaceFlux(tFlux, aState, tElementOrds, tNodeOrds);

    Kokkos::parallel_for("project to surface", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)
    {
      auto tElementOrdinal = tElementOrds(aSideOrdinal);

      Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
      for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
      {
          tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
      }

      auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
      auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
      auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
      auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

      ResultScalarType tSurfaceArea(0.0);
      surfaceArea(tElementOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tSurfaceArea);
      tSurfaceArea *= aScale;
      tSurfaceArea *= tCubatureWeight;

      // project into aResult workset
      for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
      {
          auto tElementDofOrdinal = tLocalNodeOrds[tNode] * DofsPerNode + DofOffset;
          ResultScalarType tResult = tBasisValues(tNode)*tFlux(aSideOrdinal, aPointOrdinal)*tSurfaceArea;
          Kokkos::atomic_add(&aResult(tElementOrdinal,tElementDofOrdinal), tResult);
      }
    });
}
// class StefanBoltzmann::operator()

}
// namespace Plato
