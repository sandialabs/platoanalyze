#pragma once

#include "PlatoMesh.hpp"
#include "PlatoMathTypes.hpp"

#include <Teuchos_RCPDecl.hpp>

namespace Plato
{

/******************************************************************************//**
* \brief functor that provides mesh-local node ordinal
* \param [in] aMesh Plato abstract mesh
**********************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NodesPerCell=SpaceDim+1>
class NodeOrdinal
{
  public:
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    NodeOrdinal(
      Plato::Mesh aMesh ) :
      mCells2nodes(aMesh->Connectivity()) {}

    /******************************************************************************//**
    * \brief Returns mesh-local node ordinal
    * \param [in] aCellOrdinal mesh-local element ordinal
    * \param [in] aNodeOrdinal elem-local node ordinal
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::OrdinalType aNodeOrdinal
    ) const
    {
        return mCells2nodes(aCellOrdinal*NodesPerCell + aNodeOrdinal);
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode, Plato::OrdinalType NodesPerCell=SpaceDim+1>
class VectorEntryOrdinal
{
  public:
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    VectorEntryOrdinal(
      Plato::Mesh mesh ) :
      mCells2nodes(mesh->Connectivity()) {}

    KOKKOS_INLINE_FUNCTION Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType nodeOrdinal, Plato::OrdinalType dofOrdinal=0) const
    {
        Plato::OrdinalType vertexNumber = mCells2nodes(cellOrdinal*NodesPerCell + nodeOrdinal);
        return vertexNumber * DofsPerNode + dofOrdinal;
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NodesPerCell>
class NodeCoordinate
{
  private:
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;
    const Plato::ScalarVectorT<const Plato::Scalar> mCoords;

  public:
    NodeCoordinate(Plato::Mesh aMesh) :
      mCells2nodes(aMesh->Connectivity()),
      mCoords(aMesh->Coordinates())
      {
        if (aMesh->NumDimensions() != SpaceDim || aMesh->NumNodesPerElement() != NodesPerCell)
        {
            throw std::runtime_error("Input mesh doesn't match physics spatial dimension and/or nodes per cell.");
        }
      }

    KOKKOS_INLINE_FUNCTION
    Plato::Scalar
    operator()(Plato::OrdinalType aCellOrdinal, Plato::OrdinalType aNodeOrdinal, Plato::OrdinalType aDimOrdinal) const
    {
        const Plato::OrdinalType tVertexNumber = mCells2nodes(aCellOrdinal*NodesPerCell + aNodeOrdinal);
        const Plato::Scalar tCoord = mCoords(tVertexNumber * SpaceDim + aDimOrdinal);
        return tCoord;
    }
};
/******************************************************************************/

/******************************************************************************/
/*! InertialForces Functor.
*
*   Evaluates cell inertial forces.
*/
/******************************************************************************/
class ComputeProjectionWorkset
{
public:
    /******************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarVectorT<Plato::Scalar> & tBasisFunctions,
                                       const Plato::ScalarMultiVectorT<GaussPointScalarType> & aStateValues,
                                       const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult,
                                             Plato::Scalar scale = 1.0 ) const
    /******************************************************************************/
    {
        const Plato::OrdinalType tNumNodesPerCell = tBasisFunctions.size();
        const Plato::OrdinalType tNumDofsPerNode = aStateValues.extent(1);
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofsPerNode; tDofIndex++)
            {
                Plato::OrdinalType tMyDofIndex = (tNumDofsPerNode * tNodeIndex) + tDofIndex;
                aResult(aCellOrdinal, tMyDofIndex) += scale * tBasisFunctions(tNodeIndex)
                        * aStateValues(aCellOrdinal, tDofIndex) * aCellVolume(aCellOrdinal);
            }
        }
    }
};
/******************************************************************************/

/******************************************************************************/
template<Plato::OrdinalType NodesPerCell,
         Plato::OrdinalType DofsPerNode_I,
         Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
class BlockMatrixTransposeEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVectorT mRowMap;
    const typename CrsMatrixType::OrdinalVectorT mColumnIndices;
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    BlockMatrixTransposeEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> matrix, Plato::Mesh mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->Connectivity()) { }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()
    (Plato::OrdinalType cellOrdinal, 
     Plato::OrdinalType icellDof, 
     Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode_I;
        auto iDof  = icellDof % DofsPerNode_I;
        auto jNode = jcellDof / DofsPerNode_J;
        auto jDof  = jcellDof % DofsPerNode_J;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes(cellOrdinal * NodesPerCell + iNode);
        Plato::OrdinalType jLocalOrdinal = mCells2nodes(cellOrdinal * NodesPerCell + jNode);
        return this->getEntryOrdinal(iLocalOrdinal, jLocalOrdinal, iDof, jDof);
    }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()
    (Plato::OrdinalType icellOrdinal, 
     Plato::OrdinalType jcellOrdinal, 
     Plato::OrdinalType icellDof, 
     Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode_I;
        auto iDof  = icellDof % DofsPerNode_I;
        auto jNode = jcellDof / DofsPerNode_J;
        auto jDof  = jcellDof % DofsPerNode_J;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes(icellOrdinal * NodesPerCell + iNode);
        Plato::OrdinalType jLocalOrdinal = mCells2nodes(jcellOrdinal * NodesPerCell + jNode);
        return this->getEntryOrdinal(iLocalOrdinal, jLocalOrdinal, iDof, jDof);
    }

  private:
    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    getEntryOrdinal
    (Plato::OrdinalType iLocalOrdinal, 
     Plato::OrdinalType jLocalOrdinal,
     Plato::OrdinalType iDof,
     Plato::OrdinalType jDof) const
    {
        Plato::OrdinalType rowStart = mRowMap(jLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(jLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == iLocalOrdinal)
          {
            return entryOrdinal*DofsPerNode_I*DofsPerNode_J+jDof*DofsPerNode_I+iDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};

/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType BlockSize_I, Plato::OrdinalType BlockSize_J>
class LocalByGlobalEntryFunctor
{
  private:
    const typename CrsMatrixType::RowMapVectorT mRowMap;
    const typename CrsMatrixType::OrdinalVectorT mColumnIndices;
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

    using MatrixT = Teuchos::RCP<Plato::CrsMatrixType>;
    using MeshT   = Plato::Mesh;

  public:
    LocalByGlobalEntryFunctor(
        MatrixT tMatrix,
        MeshT   tMesh
    ) :
      mRowMap        (tMatrix->rowMap()),
      mColumnIndices (tMatrix->columnIndices()),
      mCells2nodes   (tMesh->Connectivity()) { }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()(Plato::OrdinalType cellOrdinal, Plato::OrdinalType icellDof, Plato::OrdinalType jcellDof) const
    {
        auto jNode = jcellDof / BlockSize_J;
        auto jDof  = jcellDof % BlockSize_J;
        Plato::OrdinalType jLocalOrdinal = mCells2nodes(cellOrdinal * (SpaceDim+1) + jNode);
        Plato::OrdinalType rowStart = mRowMap(cellOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(cellOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*BlockSize_I*BlockSize_J + icellDof*BlockSize_J + jDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};

/******************************************************************************/
template<typename ElementType>
class GlobalByLocalEntryFunctor
{
    const typename CrsMatrixType::RowMapVectorT mRowMap;
    const typename CrsMatrixType::OrdinalVectorT mColumnIndices;
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    GlobalByLocalEntryFunctor(Teuchos::RCP<Plato::CrsMatrixType> matrix, Plato::Mesh mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->Connectivity()) { }

    /******************************************************************************//**
    * \brief Returns offset into global by local block matrix.  Matrix is assumed to 
    *        be [N,G] where N is the number of nodes in the mesh, and G is the number
    *        of gauss points in the mesh.  The block size is [d,s] where d is the number
    *        of dofs per node and s is the number of states per gauss point.
    * \param [in] aCellOrdinal mesh-local element ordinal
    * \param [in] aGpOrdinal elem-local gauss point ordinal
    * \param [in] aICellDofOrdinal ordinal into array of elem-node dofs, e.g., [n1_ux, n1_uy, n1_uz, n2_ux, ...]
    * \param [in] aIGpStateOrdinal ordinal into array of gp states, e.g., [gp_1, gp_2, ..., gp_s]
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Plato::OrdinalType aGpOrdinal,
        Plato::OrdinalType aICellDofOrdinal,
        Plato::OrdinalType aIGpStateOrdinal
    ) const
    {
        auto iNode = aICellDofOrdinal / ElementType::mNumDofsPerNode;
        auto iDof  = aICellDofOrdinal % ElementType::mNumDofsPerNode;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes(aCellOrdinal * ElementType::mNumNodesPerCell + iNode);
        Plato::OrdinalType jLocalOrdinal = aCellOrdinal*ElementType::mNumGaussPoints + aGpOrdinal;
        Plato::OrdinalType rowStart = mRowMap(iLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(iLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*ElementType::mNumDofsPerNode*ElementType::mNumLocalStatesPerGP+iDof*ElementType::mNumLocalStatesPerGP+aIGpStateOrdinal;
          }
        }
        return Plato::OrdinalType(-1);
    }
};

/******************************************************************************/
template<Plato::OrdinalType NodesPerCell,
         Plato::OrdinalType DofsPerNode_I,
         Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
class BlockMatrixEntryOrdinal
{
  private:
    const typename CrsMatrixType::RowMapVectorT mRowMap;
    const typename CrsMatrixType::OrdinalVectorT mColumnIndices;
    const Plato::OrdinalVectorT<const Plato::OrdinalType> mCells2nodes;

  public:
    BlockMatrixEntryOrdinal(Teuchos::RCP<Plato::CrsMatrixType> matrix, Plato::Mesh mesh ) :
      mRowMap(matrix->rowMap()),
      mColumnIndices(matrix->columnIndices()),
      mCells2nodes(mesh->Connectivity()) { }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()
    (Plato::OrdinalType cellOrdinal, 
     Plato::OrdinalType icellDof, 
     Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode_I;
        auto iDof  = icellDof % DofsPerNode_I;
        auto jNode = jcellDof / DofsPerNode_J;
        auto jDof  = jcellDof % DofsPerNode_J;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes(cellOrdinal * NodesPerCell + iNode);
        Plato::OrdinalType jLocalOrdinal = mCells2nodes(cellOrdinal * NodesPerCell + jNode);
        return this->getEntryOrdinal(iLocalOrdinal, jLocalOrdinal, iDof, jDof);
    }

    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    operator()
    (Plato::OrdinalType icellOrdinal, 
     Plato::OrdinalType jcellOrdinal, 
     Plato::OrdinalType icellDof, 
     Plato::OrdinalType jcellDof) const
    {
        auto iNode = icellDof / DofsPerNode_I;
        auto iDof  = icellDof % DofsPerNode_I;
        auto jNode = jcellDof / DofsPerNode_J;
        auto jDof  = jcellDof % DofsPerNode_J;
        Plato::OrdinalType iLocalOrdinal = mCells2nodes(icellOrdinal * NodesPerCell + iNode);
        Plato::OrdinalType jLocalOrdinal = mCells2nodes(jcellOrdinal * NodesPerCell + jNode);
        return this->getEntryOrdinal(iLocalOrdinal, jLocalOrdinal, iDof, jDof);
    }

  private:
    KOKKOS_INLINE_FUNCTION
    Plato::OrdinalType
    getEntryOrdinal
    (Plato::OrdinalType iLocalOrdinal, 
     Plato::OrdinalType jLocalOrdinal,
     Plato::OrdinalType iDof,
     Plato::OrdinalType jDof) const
    {
        Plato::OrdinalType rowStart = mRowMap(iLocalOrdinal);
        Plato::OrdinalType rowEnd   = mRowMap(iLocalOrdinal+1);
        for (Plato::OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
        {
          if (mColumnIndices(entryOrdinal) == jLocalOrdinal)
          {
            return entryOrdinal*DofsPerNode_I*DofsPerNode_J+iDof*DofsPerNode_J+jDof;
          }
        }
        return Plato::OrdinalType(-1);
    }
};
/******************************************************************************/

} // end namespace Plato
