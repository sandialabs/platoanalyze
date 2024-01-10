//
//  CrsMatrix.hpp
//
//
//  Created by Roberts, Nathan V on 6/20/17.
//
//

#ifndef PLATO_CRS_MATRIX_HPP
#define PLATO_CRS_MATRIX_HPP

#include "PlatoTypes.hpp"

#include <Kokkos_Core.hpp>

namespace Plato {

/// @brief Checks that the row, column, and entry arrays @a aRowMap, @a aColMap, and @a aEntries
/// have the expected sizes.
///
/// It is expected that the last entry in the row map is the number of non-zero blocks.
/// The size of the column map should match the number of non-zero blocks, and the size
/// of the entries array should be the number of non-zero blocks times the number of
/// entries per block given by arguments @a aNumRowsPerBlock and @a aNumColsPerBlock.
template<typename Ordinal>
bool validCrsMapSizes(
  const Kokkos::View<Ordinal*, MemSpace> aRowMap,
  const Kokkos::View<Ordinal*,  MemSpace> aColMap,
  const Kokkos::View<Scalar*, MemSpace> aEntries,
  const int aNumRowsPerBlock = 1,
  const int aNumColsPerBlock = 1 )
{
  if(aRowMap.size() == 0 && aColMap.size() == 0 && aEntries.size() == 0)
  {
    // All maps have zero entries
    return true;
  }
  else
  {
    Ordinal tNNZFromRowMap = 0;
    // Last entry in the row map is expected to be the number of non-zero blocks
    Kokkos::deep_copy(tNNZFromRowMap, Kokkos::subview(aRowMap, aRowMap.extent(0) - 1));
    // Check that the last entry in the row map matches the number of column indices
    const bool tColumnsMatch = tNNZFromRowMap == aColMap.extent(0);
    // Check that the last entry matches number of values times block size
    const Ordinal tExpectedNumEntries = tNNZFromRowMap * aNumRowsPerBlock * aNumColsPerBlock;
    const bool tEntriesMatch = tExpectedNumEntries == aEntries.extent(0);
    return tColumnsMatch && tEntriesMatch;
  }
}

template < class Ordinal = Plato::OrdinalType >
class CrsMatrix {
 public:
  typedef Kokkos::View<Ordinal*, MemSpace> OrdinalVectorT;
  typedef Kokkos::View<Scalar*,  MemSpace> ScalarVectorT;
  typedef Kokkos::View<Ordinal*, MemSpace> RowMapVectorT;

 private:

  RowMapVectorT  mRowMap;
  OrdinalVectorT mColumnIndices;
  ScalarVectorT  mEntries;

  int  mNumRows = -1;
  int  mNumCols = -1;
  int  mNumRowsPerBlock = 1;
  int  mNumColsPerBlock = 1;
  bool mIsBlockMatrix = false;

 public:
  decltype(mIsBlockMatrix)  isBlockMatrix()     const { return mIsBlockMatrix; }
  decltype(mNumRowsPerBlock) numRowsPerBlock()  const { return mNumRowsPerBlock; }
  decltype(mNumColsPerBlock) numColsPerBlock()  const { return mNumColsPerBlock; }

  decltype(mNumRows) numRows() const
  { return (mNumRows != -1) ? mNumRows : throw std::logic_error("requested unset value"); }

  decltype(mNumCols) numCols() const
  { return (mNumCols != -1) ? mNumCols : throw std::logic_error("requested unset value"); }

  CrsMatrix() = default;

  CrsMatrix( int           aNumRows,
             int           aNumCols,
             int           aNumRowsPerBlock,
             int           aNumColsPerBlock
           ) :
            mNumRows         (aNumRows),
            mNumCols         (aNumCols),
            mNumRowsPerBlock (aNumRowsPerBlock),
            mNumColsPerBlock (aNumColsPerBlock),
            mIsBlockMatrix   (mNumColsPerBlock*mNumRowsPerBlock > 1) {}

  CrsMatrix( RowMapVectorT  aRowMap,
             OrdinalVectorT aColIndices,
             ScalarVectorT  aEntries,
             int            aNumRowsPerBlock=1,
             int            aNumColsPerBlock=1
           ) :
            mRowMap          (aRowMap),
            mColumnIndices   (aColIndices),
            mEntries         (aEntries),
            mNumRows         (-1),
            mNumCols         (-1),
            mNumRowsPerBlock (aNumRowsPerBlock),
            mNumColsPerBlock (aNumColsPerBlock),
            mIsBlockMatrix   (mNumColsPerBlock*mNumRowsPerBlock > 1)
          {
            assert(validCrsMapSizes<Ordinal>(aRowMap, aColIndices, aEntries, aNumRowsPerBlock, aNumColsPerBlock));
          }

  CrsMatrix( RowMapVectorT  aRowMap,
             OrdinalVectorT aColIndices,
             ScalarVectorT  aEntries,
             int            aNumRows,
             int            aNumCols,
             int            aNumRowsPerBlock,
             int            aNumColsPerBlock
          ) :
            mRowMap          (aRowMap),
            mColumnIndices   (aColIndices),
            mEntries         (aEntries),
            mNumRows         (aNumRows),
            mNumCols         (aNumCols),
            mNumRowsPerBlock (aNumRowsPerBlock),
            mNumColsPerBlock (aNumColsPerBlock),
            mIsBlockMatrix   (mNumColsPerBlock*mNumRowsPerBlock > 1)
          {
            assert(validCrsMapSizes<Ordinal>(aRowMap, aColIndices, aEntries, aNumRowsPerBlock, aNumColsPerBlock));
          }

  KOKKOS_INLINE_FUNCTION decltype(mRowMap)        rowMap()        { return mRowMap; }
  KOKKOS_INLINE_FUNCTION decltype(mColumnIndices) columnIndices() { return mColumnIndices; }
  KOKKOS_INLINE_FUNCTION decltype(mEntries)       entries()       { return mEntries; }

  KOKKOS_INLINE_FUNCTION const decltype(mRowMap)        rowMap()        const { return mRowMap; }
  KOKKOS_INLINE_FUNCTION const decltype(mColumnIndices) columnIndices() const { return mColumnIndices; }
  KOKKOS_INLINE_FUNCTION const decltype(mEntries)       entries()       const { return mEntries; }

  KOKKOS_INLINE_FUNCTION void setRowMap       (decltype(mRowMap)        aRowMap)        { mRowMap = aRowMap; }
  KOKKOS_INLINE_FUNCTION void setColumnIndices(decltype(mColumnIndices) aColumnIndices) { mColumnIndices = aColumnIndices; }
  KOKKOS_INLINE_FUNCTION void setEntries      (decltype(mEntries)       aEntries)       { mEntries = aEntries; }

};

}  // namespace Plato

#endif /* CrsMatrix_h */
