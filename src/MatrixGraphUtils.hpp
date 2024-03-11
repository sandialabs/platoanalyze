#pragma once

#include "PlatoMesh.hpp"
#include "PlatoMathTypes.hpp"

#include "SpatialModel.hpp"

#include <Teuchos_RCPDecl.hpp>

namespace Plato
{

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param aMesh Plato abstract mesh on which the matrix is based.

  Create a block matrix from connectivity in mesh with block size
  DofsPerNode_I X DofsPerElem_J.

  This function creates a matrix that stores a transpose of the gradient of
  local element states wrt nodal degrees of freedom.  Each column has the same
  number of non-zero block entries (NNodesPerCell)
*/
template <typename MatrixType,
          typename ElementType>
Teuchos::RCP<MatrixType>
CreateGlobalByLocalBlockMatrix( Plato::Mesh aMesh )
/******************************************************************************/
{
    Plato::OrdinalVectorT<const Plato::OrdinalType> tOffsetMap;
    Plato::OrdinalVectorT<const Plato::OrdinalType> tElementOrds;
    aMesh->NodeElementGraph(tOffsetMap, tElementOrds);

    auto tNumElems = aMesh->NumElements();
    auto tNumNodes = aMesh->NumNodes();
    auto tNumNonZeros = tElementOrds.extent(0)*ElementType::mNumGaussPoints;

    constexpr Plato::OrdinalType numBlockDofs = ElementType::mNumDofsPerNode*ElementType::mNumLocalStatesPerGP;

    typename MatrixType::RowMapVectorT  rowMap        ("row map",        tNumNodes+1);
    typename MatrixType::ScalarVectorT  entries       ("matrix entries", tNumNonZeros*numBlockDofs);
    typename MatrixType::OrdinalVectorT columnIndices ("column indices", tNumNonZeros);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
    {
      auto tFrom = tOffsetMap(aNodeOrdinal);
      auto tTo   = tOffsetMap(aNodeOrdinal+1);
      rowMap(aNodeOrdinal)   = ElementType::mNumGaussPoints*tFrom;
      rowMap(aNodeOrdinal+1) = ElementType::mNumGaussPoints*tTo;

      for( decltype(tFrom) tOffset = tFrom; tOffset < tTo; tOffset++ )
      {
          for( decltype(tFrom) tGPOrd = 0; tGPOrd < ElementType::mNumGaussPoints; tGPOrd++ )
          {
              auto tColumnEntry = ElementType::mNumGaussPoints * tOffset + tGPOrd;
              columnIndices(tColumnEntry) = ElementType::mNumGaussPoints*tElementOrds(tOffset) + tGPOrd;
          }
      }
    });

    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries,
                     tNumNodes*ElementType::mNumDofsPerNode,
                     tNumElems*ElementType::mNumGaussPoints*ElementType::mNumLocalStatesPerGP,
                     ElementType::mNumDofsPerNode,
                     ElementType::mNumLocalStatesPerGP )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  @param aMesh Plato abstract mesh on which the matrix is based.

  Create a block matrix from connectivity in mesh with block size
  DofsPerElem_I X DofsPerNode_J.

  This function creates a matrix that stores a gradient of local element
  states wrt nodal degrees of freedom.  Each row has the same number of
  non-zero block entries (NNodesPerCell)
*/
template <typename MatrixType,
          Plato::OrdinalType NodesPerElem,
          Plato::OrdinalType DofsPerElem_I,
          Plato::OrdinalType DofsPerNode_J>
Teuchos::RCP<MatrixType>
CreateLocalByGlobalBlockMatrix( Plato::Mesh aMesh )
/******************************************************************************/
{
    const auto& mCells2nodes = aMesh->Connectivity();

    auto tNumElems = aMesh->NumElements();
    auto tNumNonZeros = tNumElems*NodesPerElem;

    constexpr Plato::OrdinalType numBlockDofs = DofsPerElem_I*DofsPerNode_J;

    typename MatrixType::RowMapVectorT  rowMap        ("row map",        tNumElems+1);
    typename MatrixType::ScalarVectorT  entries       ("matrix entries", tNumNonZeros*numBlockDofs);
    typename MatrixType::OrdinalVectorT columnIndices ("column indices", tNumNonZeros);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumElems), KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
      auto tFrom = aCellOrdinal*NodesPerElem;
      auto tTo   = (aCellOrdinal+1)*NodesPerElem;
      rowMap(aCellOrdinal)   = tFrom;
      rowMap(aCellOrdinal+1) = tTo;

      decltype(aCellOrdinal) tLocalIndex = 0;
      for( decltype(tFrom) tColumnEntry = tFrom; tColumnEntry < tTo; tColumnEntry++ )
      {
          columnIndices(tColumnEntry) = mCells2nodes(aCellOrdinal*NodesPerElem + tLocalIndex++);
      }
    });

    auto tNumNodes = aMesh->NumNodes();
    auto retMatrix = Teuchos::rcp(
     new MatrixType( rowMap, columnIndices, entries,
                     tNumElems*DofsPerElem_I, tNumNodes*DofsPerNode_J,
                     DofsPerElem_I, DofsPerNode_J )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix of type MatrixType

  \param mesh Plato abstract mesh on which the matrix is based.  

  Create a block matrix from connectivity in mesh with block size
  DofsPerNode_I X DofsPerNode_J.
*/
template <typename MatrixType, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
Teuchos::RCP<MatrixType>
CreateBlockMatrix( const Plato::SpatialModel & aSpatialModel )
/******************************************************************************/
{
    Plato::OrdinalVector tOffsetMap;
    Plato::OrdinalVector tNodeOrds;
    aSpatialModel.NodeNodeGraph(tOffsetMap, tNodeOrds);

    auto numRows = tOffsetMap.size() - 1;
    auto nnz = tNodeOrds.size();
    constexpr Plato::OrdinalType numBlockDofs = DofsPerNode_I*DofsPerNode_J;
    typename MatrixType::ScalarVectorT entries("matrix entries", nnz*numBlockDofs);
    auto retMatrix = Teuchos::rcp(
     new MatrixType( tOffsetMap, tNodeOrds, entries,
                     numRows*DofsPerNode_I, numRows*DofsPerNode_J,
                     DofsPerNode_I, DofsPerNode_J )
    );
    return retMatrix;
}

/******************************************************************************/
/*!
  \brief Create a matrix transpose of type MatrixType

  \param mesh Plato abstract mesh on which the matrix is based.  

  Create a block matrix from connectivity in mesh with block size
  DofsPerNode_J X DofsPerNode_I.
*/
template <typename MatrixType, Plato::OrdinalType DofsPerNode_I, Plato::OrdinalType DofsPerNode_J=DofsPerNode_I>
Teuchos::RCP<MatrixType>
CreateBlockMatrixTranspose( const Plato::SpatialModel & aSpatialModel )
/******************************************************************************/
{
    Plato::OrdinalVector tOffsetMap;
    Plato::OrdinalVector tNodeOrds;
    aSpatialModel.NodeNodeGraphTranspose(tOffsetMap, tNodeOrds);

    auto numRows = tOffsetMap.size() - 1;
    auto nnz = tNodeOrds.size();
    constexpr Plato::OrdinalType numBlockDofs = DofsPerNode_I*DofsPerNode_J;
    typename MatrixType::ScalarVectorT entries("matrix entries", nnz*numBlockDofs);
    auto retMatrix = Teuchos::rcp(
     new MatrixType( tOffsetMap, tNodeOrds, entries,
                     numRows*DofsPerNode_J, numRows*DofsPerNode_I,
                     DofsPerNode_J, DofsPerNode_I )
    );
    return retMatrix;
}

} // end namespace Plato
