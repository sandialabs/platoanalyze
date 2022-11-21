#pragma once

#include "FadTypes.hpp"
#include "SpatialModel.hpp"
#include "SurfaceArea.hpp"
#include "ExpressionEvaluator.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Class for the evaluation of natural boundary condition surface integrals
 * of type: STATE_FUNCTION
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
class SurfaceStateIntegral
{
private:
    const std::string mSideSetName; /*!< side set name */
    const std::vector<std::string> mFluxExpressions;
    const std::vector<std::string> mStateNames;

public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    SurfaceStateIntegral(
        const std::string & aSideSetName,
        const std::vector<std::string>& aFlux,
        const std::vector<std::string>& aStateNames);

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
    void evaluateSurfaceExpressions(
      Plato::ScalarArray3DT<StateScalarType> tFlux,
      Plato::ScalarMultiVectorT<StateScalarType> tState,
      Plato::OrdinalVectorT<const Plato::OrdinalType> tSideSetElementOrds,
      Plato::OrdinalVectorT<const Plato::OrdinalType> tSideSetLocalNodeOrds) const;

}; // class SurfaceStateIntegral

/***************************************************************************//**
 * \brief SurfaceStateIntegral::SurfaceStateIntegral constructor definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
SurfaceStateIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::SurfaceStateIntegral(
  const std::string & aSideSetName,
  const std::vector<std::string>& aFlux,
  const std::vector<std::string>& aStateNames
) :
    mSideSetName(aSideSetName),
    mFluxExpressions(aFlux),
    mStateNames(aStateNames)
{
}

template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType>
void SurfaceStateIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::evaluateSurfaceExpressions(
      Plato::ScalarArray3DT<StateScalarType> aFlux,
      Plato::ScalarMultiVectorT<StateScalarType> aState,
      Plato::OrdinalVectorT<const Plato::OrdinalType> aSideSetElementOrds,
      Plato::OrdinalVectorT<const Plato::OrdinalType> aSideSetLocalNodeOrds
) const
{
    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tCubaturePoints  = ElementType::Face::getCubPoints();
    auto tNumPoints = tCubatureWeights.size();
    Plato::OrdinalType tNumFaces = aSideSetElementOrds.size();

    Plato::ScalarVectorT<StateScalarType> tStateAtGPs("GP state", tNumFaces*tNumPoints);
    Plato::ScalarMultiVectorT<StateScalarType> tExpressionValues("expression values", tNumFaces*tNumPoints, 1);

    for(int iDof=0; iDof<NumDofs; iDof++)
    {
        ExpressionEvaluator<Plato::ScalarMultiVectorT<StateScalarType>,
                            Plato::ScalarMultiVectorT<StateScalarType>,
                            Plato::ScalarVectorT<StateScalarType>,
                            Plato::Scalar> tExpEval;

        tExpEval.parse_expression(mFluxExpressions[iDof].c_str());
        tExpEval.setup_storage(tNumFaces*tNumPoints, 1);

        Kokkos::parallel_for("compute state", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tElementOrdinal = aSideSetElementOrds(aSideOrdinal);
            auto tCubPoint = tCubaturePoints(iGpOrdinal);
            auto tBasisValues = ElementType::Face::basisValues(tCubPoint);
        
            StateScalarType tInterpolatedValue(0.0);
            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
            {
                auto tLocalNodeOrd = aSideSetLocalNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace+tNode);
                tInterpolatedValue += tBasisValues(tNode)*aState(tElementOrdinal, DofsPerNode*tLocalNodeOrd+DofOffset+iDof);
            }
            auto tEntryOrdinal = aSideOrdinal*tNumPoints + iGpOrdinal;
            tStateAtGPs(tEntryOrdinal) = tInterpolatedValue;
        });

        tExpEval.set_variable(mStateNames[iDof].c_str(), tStateAtGPs);
        Kokkos::parallel_for("evaluate expression", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tEntryOrdinal = aSideOrdinal*tNumPoints + iGpOrdinal;

            tExpEval.evaluate_expression( tEntryOrdinal, tExpressionValues );
        });
        Kokkos::fence();
        Kokkos::parallel_for("evaluate expression", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType aSideOrdinal, const Plato::OrdinalType iGpOrdinal)
        {
            auto tEntryOrdinal = aSideOrdinal*tNumPoints + iGpOrdinal;

            aFlux(aSideOrdinal, iGpOrdinal, iDof) = tExpressionValues(tEntryOrdinal, 0);
        });
        tExpEval.clear_storage();
    }
}


/***************************************************************************//**
 * \brief SurfaceStateIntegral::operator() function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void SurfaceStateIntegral<ElementType, NumDofs, DofsPerNode, DofOffset>::operator()(
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

    Plato::ScalarArray3DT<StateScalarType> tFlux("fluxes", tNumFaces, tNumPoints, NumDofs);
    evaluateSurfaceExpressions(tFlux, aState, tElementOrds, tNodeOrds);

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
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
          for( Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
          {
              auto tElementDofOrdinal = tLocalNodeOrds[tNode] * DofsPerNode + tDof + DofOffset;
              ResultScalarType tResult = tBasisValues(tNode)*tFlux(aSideOrdinal, aPointOrdinal, tDof)*tSurfaceArea;
              Kokkos::atomic_add(&aResult(tElementOrdinal,tElementDofOrdinal), tResult);
          }
      }
    }, "surface load integral");
}
// class SurfaceStateIntegral::operator()

}
// namespace Plato
