#pragma once

#include "PlatoStaticsTypes.hpp"
#include "SpatialModel.hpp"
#include "SurfaceArea.hpp"
#include "contact/AbstractSurfaceDisplacement.hpp"
#include "contact/AbstractContactForce.hpp"

namespace Plato
{

namespace Contact
{

template<typename EvaluationType,
         Plato::OrdinalType NumDofsPerNode = EvaluationType::ElementType::mNumSpatialDims>
class IntegrateContactForce
{
private:
    using ElementType = typename EvaluationType::ElementType;
    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

public:
    IntegrateContactForce
    (const Plato::SpatialModel                                       & aSpatialModel,
     const std::string                                               & aSideSet,
           Teuchos::RCP<AbstractSurfaceDisplacement<EvaluationType>>   aComputeSurfaceDisp,
           Teuchos::RCP<AbstractContactForce<EvaluationType>>          aComputeContactForce) :
     mComputeSurfaceDisplacement(aComputeSurfaceDisp),
     mComputeContactForce(aComputeContactForce)
    {
        mElementOrds = aSpatialModel.Mesh->GetSideSetElements(aSideSet);
        mLocalNodeOrds = aSpatialModel.Mesh->GetSideSetLocalNodes(aSideSet);
    }

    void operator()
     (const Plato::ScalarMultiVectorT <StateScalarType> & aState,
     const Plato::ScalarArray3DT     <ConfigScalarType> & aConfig,
           Plato::ScalarMultiVectorT <ResultScalarType> & aResult,
           Plato::Scalar                                  aTimeStep) const
    {
        Plato::OrdinalType tNumFaces = mElementOrds.size();
        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tCubatureWeights = ElementType::Face::getCubWeights();
        auto tNumPoints = tCubatureWeights.size();

        Plato::SurfaceArea<ElementType> surfaceArea;

        Plato::ScalarArray3DT<StateScalarType> tSurfaceDisplacement("displacement on contact surface", tNumFaces, tNumPoints, NumDofsPerNode);
        (*mComputeSurfaceDisplacement)(mElementOrds, aState, tSurfaceDisplacement);

        Plato::ScalarArray3DT<ResultScalarType> tContactForce("contact force at cubature points", tNumFaces, tNumPoints, NumDofsPerNode);
        (*mComputeContactForce)(mElementOrds, mLocalNodeOrds, tSurfaceDisplacement, aConfig, tContactForce);

        auto tElementOrds = mElementOrds;
        auto tLocalNodeOrds = mLocalNodeOrds;
        Kokkos::parallel_for("project contact force to nodes", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumFaces, tNumPoints}),
        KOKKOS_LAMBDA(const Plato::OrdinalType & iCellOrdinal, const Plato::OrdinalType & iGPOrdinal)
        {
            auto tCellOrdinal = tElementOrds(iCellOrdinal);
            auto tCubaturePoint = tCubaturePoints(iGPOrdinal);
            auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
            auto tBasisGrads = ElementType::Face::basisGrads(tCubaturePoint);

            Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodes;
            for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
            {
                tLocalNodes(tNodeOrd) = tLocalNodeOrds(iCellOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
            }

            ResultScalarType tSurfaceArea(0.0);
            surfaceArea(tCellOrdinal, tLocalNodes, tBasisGrads, aConfig, tSurfaceArea);
            tSurfaceArea *= tCubatureWeights(iGPOrdinal);

            for( Plato::OrdinalType tNode=0; tNode<ElementType::mNumNodesPerFace; tNode++)
            {
                for( Plato::OrdinalType tDof=0; tDof<NumDofsPerNode; tDof++)
                {
                    auto tElementDofOrdinal = tLocalNodes(tNode) * NumDofsPerNode + tDof;
                    ResultScalarType tResult = tSurfaceArea*tBasisValues(tNode)*tContactForce(iCellOrdinal, iGPOrdinal, tDof);
                    Kokkos::atomic_add(&aResult(tCellOrdinal, tElementDofOrdinal), tResult);
                }
            }

        });

    }

private:
    Teuchos::RCP<AbstractSurfaceDisplacement<EvaluationType>> mComputeSurfaceDisplacement;
    Teuchos::RCP<AbstractContactForce<EvaluationType>> mComputeContactForce;
    Plato::OrdinalVectorT<const Plato::OrdinalType> mElementOrds;
    Plato::OrdinalVectorT<const Plato::OrdinalType> mLocalNodeOrds;

};

}

}
