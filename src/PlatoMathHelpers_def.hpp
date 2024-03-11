#pragma once

#include "PlatoMathHelpers.hpp"

#include <sstream>
#include <cassert>

#include <Kokkos_Macros.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_SortCrs.hpp>

#include "PlatoStaticsTypes.hpp"
#include "PlatoMathFunctors.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

template<typename ScalarT>
void MatrixTimesVectorPlusVector(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                                 const Plato::ScalarVectorT<ScalarT> & aInput,
                                 const Plato::ScalarVectorT<ScalarT> & aOutput)
{
    if(aMatrix->numCols() != aInput.size())
    {
        std::ostringstream tMsg;
        tMsg << "DIMENSION MISMATCH.  INPUT VECTOR LENGTH DOES NOT MATCH THE NUMBER OF COLUMNS IN MATRIX A.  "
             << "INPUT VECTOR LENGTH = '" <<  aInput.size() << "' AND THE NUMBER OF COLUMNS IN MATRIX A = '"
             << aMatrix->numCols() << "'. INPUT VECTOR LABEL IS '" << aInput.label()
             << "' AND OUTPUT VECTOR LABEL IS '" << aOutput.label() << "'.";
        ANALYZE_THROWERR(tMsg.str());
    }
    if(aMatrix->numRows() != aOutput.size())
    {
        std::ostringstream tMsg;
        tMsg << "DIMENSION MISMATCH.  OUTPUT VECTOR LENGTH DOES NOT MATCH THE NUMBER OF ROWS IN MATRIX A.  "
             << "OUTPUT VECTOR LENGTH = '" <<  aOutput.size() << "' AND THE NUMBER OF ROWS IN MATRIX A = '"
             << aMatrix->numRows() << "'. OUTPUT VECTOR LABEL IS '" << aOutput.label() << "'.";
        ANALYZE_THROWERR(tMsg.str());
    }

    if(aMatrix->isBlockMatrix())
    {
        auto tNodeRowMap = aMatrix->rowMap();
        auto tNodeColIndices = aMatrix->columnIndices();
        auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
        auto tNumColsPerBlock = aMatrix->numColsPerBlock();
        auto tEntries = aMatrix->entries();
        auto tNumNodeRows = tNodeRowMap.size() - 1;

        Kokkos::parallel_for("BlockMatrix * Vector_a + Vector_b", Kokkos::RangePolicy<>(0, tNumNodeRows), KOKKOS_LAMBDA(const Plato::OrdinalType & aNodeRowOrdinal)
        {
            auto tRowStartIndex = tNodeRowMap(aNodeRowOrdinal);
            auto tRowEndIndex = tNodeRowMap(aNodeRowOrdinal + 1);
            for (auto tCrsIndex = tRowStartIndex; tCrsIndex < tRowEndIndex; tCrsIndex++)
            {
                auto tNodeColumnIndex = tNodeColIndices(tCrsIndex);

                auto tFromDofColIndex = tNumColsPerBlock*tNodeColumnIndex;
                auto tToDofColIndex = tFromDofColIndex + tNumColsPerBlock;

                auto tFromDofRowIndex = tNumRowsPerBlock*aNodeRowOrdinal;
                auto tToDofRowIndex = tFromDofRowIndex + tNumRowsPerBlock;

                auto tMatrixEntryIndex = tNumRowsPerBlock*tNumColsPerBlock*tCrsIndex;
                for ( auto tDofRowIndex = tFromDofRowIndex; tDofRowIndex < tToDofRowIndex; tDofRowIndex++ )
                {
                    ScalarT tSum = 0.0;
                    for ( auto tDofColIndex = tFromDofColIndex; tDofColIndex < tToDofColIndex; tDofColIndex++ )
                    {
                        tSum += tEntries(tMatrixEntryIndex) * aInput(tDofColIndex);
                        tMatrixEntryIndex += 1;
                    }
                    aOutput(tDofRowIndex) += tSum;
                }
            }
        });
    }
    else
    {
        auto tRowMap = aMatrix->rowMap();
        auto tColIndices = aMatrix->columnIndices();
        auto tEntries = aMatrix->entries();
        auto tNumRows = tRowMap.size() - 1;

        Kokkos::parallel_for("Matrix * Vector_a + Vector_b", Kokkos::RangePolicy<>(0, tNumRows), KOKKOS_LAMBDA(const Plato::OrdinalType & aRowOrdinal)
        {
            auto tRowStart = tRowMap(aRowOrdinal);
            auto tRowEnd = tRowMap(aRowOrdinal + 1);
            ScalarT tSum = 0.0;
            for (auto tEntryIndex = tRowStart; tEntryIndex < tRowEnd; tEntryIndex++)
            {
                auto tColumnIndex = tColIndices(tEntryIndex);
                tSum += tEntries(tEntryIndex) * aInput(tColumnIndex);
            }
            aOutput(aRowOrdinal) += tSum;
        });
    }
}

Plato::Scalar diagonalAveAbs(
    Plato::CrsMatrixType const & aMatrix
)
{

  if(aMatrix.isBlockMatrix())
  {
      auto tNodeRowMap = aMatrix.rowMap();
      auto tNodeColIndices = aMatrix.columnIndices();
      auto tNumRowsPerBlock = aMatrix.numRowsPerBlock();
      auto tNumColsPerBlock = aMatrix.numColsPerBlock();
      auto tEntries = aMatrix.entries();
      auto tNumNodeRows = tNodeRowMap.size() - 1;

      if(tNumRowsPerBlock != tNumColsPerBlock)
      {
        ANALYZE_THROWERR("diagonalAveAbs expects a square matrix");
      }

      Plato::Scalar tDiagonalSum(0);
      Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumNodeRows),
      KOKKOS_LAMBDA(const Plato::OrdinalType& aNodeRowOrdinal, Plato::Scalar & aUpdate)
      {
        auto tRowStartIndex = tNodeRowMap(aNodeRowOrdinal);
        auto tRowEndIndex = tNodeRowMap(aNodeRowOrdinal + 1);
        for (auto tCrsIndex = tRowStartIndex; tCrsIndex < tRowEndIndex; tCrsIndex++)
        {
          if (tNodeColIndices(tCrsIndex) == aNodeRowOrdinal)
          {
            auto tMatrixEntryOffset = tNumRowsPerBlock*tNumColsPerBlock*tCrsIndex;
            for ( auto tIndex = 0; tIndex < tNumRowsPerBlock; tIndex++ )
            {
              aUpdate += fabs(tEntries(tMatrixEntryOffset + tNumColsPerBlock*tIndex + tIndex));
            }
          }
        }
      }, tDiagonalSum);
      return tDiagonalSum/(tNumNodeRows*tNumRowsPerBlock);
  }
  else
  {
    ANALYZE_THROWERR("diagonalAveAbs not implemented for non-block matrices");
  }
}


void shiftDiagonal(
    Plato::CrsMatrixType const & aMatrix,
    Plato::Scalar                aShift)
{

  if(aMatrix.isBlockMatrix())
  {
      auto tNodeRowMap = aMatrix.rowMap();
      auto tNodeColIndices = aMatrix.columnIndices();
      auto tNumRowsPerBlock = aMatrix.numRowsPerBlock();
      auto tNumColsPerBlock = aMatrix.numColsPerBlock();
      auto tEntries = aMatrix.entries();
      auto tNumNodeRows = tNodeRowMap.size() - 1;

      if(tNumRowsPerBlock != tNumColsPerBlock)
      {
        ANALYZE_THROWERR("shiftDiagonal expects a square matrix");
      }

      Kokkos::parallel_for("shift diagonal", Kokkos::RangePolicy<>(0, tNumNodeRows), KOKKOS_LAMBDA(const Plato::OrdinalType & aNodeRowOrdinal)
      {
        auto tRowStartIndex = tNodeRowMap(aNodeRowOrdinal);
        auto tRowEndIndex = tNodeRowMap(aNodeRowOrdinal + 1);
        for (auto tCrsIndex = tRowStartIndex; tCrsIndex < tRowEndIndex; tCrsIndex++)
        {
          if (tNodeColIndices(tCrsIndex) == aNodeRowOrdinal)
          {
            auto tMatrixEntryOffset = tNumRowsPerBlock*tNumColsPerBlock*tCrsIndex;
            for ( auto tIndex = 0; tIndex < tNumRowsPerBlock; tIndex++ )
            {
              tEntries(tMatrixEntryOffset + tNumColsPerBlock*tIndex + tIndex) += aShift;
            }
          }
        }
    });
  }
  else
  {
    ANALYZE_THROWERR("shiftDiagonal not implemented for non-block matrices");
  }
}

template<typename ScalarT>
void VectorTimesMatrixPlusVector(
    const Plato::ScalarVectorT<ScalarT>      & aInput,
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVectorT<ScalarT>      & aOutput)
{
    if(aMatrix->numRows() != aInput.size())
    {
        std::ostringstream tMsg;
        tMsg << "DIMENSION MISMATCH.  INPUT VECTOR LENGTH DOES NOT MATCH THE NUMBER OF ROWS IN MATRIX A.  "
             << "INPUT VECTOR LENGTH = '" <<  aInput.size() << "' AND THE NUMBER OF ROWS IN MATRIX A = '"
             << aMatrix->numCols() << "'. INPUT VECTOR LABEL IS '" << aInput.label()
             << "' AND OUTPUT VECTOR LABEL IS '" << aOutput.label() << "'.";
        ANALYZE_THROWERR(tMsg.str());
    }
    if(aMatrix->numCols() != aOutput.size())
    {
        std::ostringstream tMsg;
        tMsg << "DIMENSION MISMATCH.  OUTPUT VECTOR LENGTH DOES NOT MATCH THE NUMBER OF COLUMNS IN MATRIX A.  "
             << "OUTPUT VECTOR LENGTH = '" <<  aOutput.size() << "' AND THE NUMBER OF COLUMNS IN MATRIX A = '"
             << aMatrix->numRows() << "'. OUTPUT VECTOR LABEL IS '" << aOutput.label() << "'.";
        ANALYZE_THROWERR(tMsg.str());
    }

    if(aMatrix->isBlockMatrix())
    {
        auto tNodeRowMap = aMatrix->rowMap();
        auto tNodeColIndices = aMatrix->columnIndices();
        auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
        auto tNumColsPerBlock = aMatrix->numColsPerBlock();
        auto tEntries = aMatrix->entries();
        auto tNumNodeRows = tNodeRowMap.size() - 1;

        Kokkos::parallel_for("Vector_a * BlockMatrix + Vector_b", Kokkos::RangePolicy<>(0, tNumNodeRows), KOKKOS_LAMBDA(const Plato::OrdinalType & aNodeRowOrdinal)
        {
            auto tRowStartIndex = tNodeRowMap(aNodeRowOrdinal);
            auto tRowEndIndex = tNodeRowMap(aNodeRowOrdinal + 1);
            for (auto tCrsIndex = tRowStartIndex; tCrsIndex < tRowEndIndex; tCrsIndex++)
            {
                auto tNodeColumnIndex = tNodeColIndices(tCrsIndex);

                auto tFromDofColIndex = tNumColsPerBlock*tNodeColumnIndex;
                auto tToDofColIndex = tFromDofColIndex + tNumColsPerBlock;

                auto tFromDofRowIndex = tNumRowsPerBlock*aNodeRowOrdinal;
                auto tToDofRowIndex = tFromDofRowIndex + tNumRowsPerBlock;

                auto tMatrixEntryIndex = tNumRowsPerBlock*tNumColsPerBlock*tCrsIndex;
                for ( auto tDofRowIndex = tFromDofRowIndex; tDofRowIndex < tToDofRowIndex; tDofRowIndex++ )
                {
                    for ( auto tDofColIndex = tFromDofColIndex; tDofColIndex < tToDofColIndex; tDofColIndex++ )
                    {
                        Kokkos::atomic_add(&aOutput(tDofColIndex), tEntries(tMatrixEntryIndex) * aInput(tDofRowIndex));
                        tMatrixEntryIndex += 1;
                    }
                }
            }
        });
    }
    else
    {
        auto tRowMap = aMatrix->rowMap();
        auto tColIndices = aMatrix->columnIndices();
        auto tEntries = aMatrix->entries();
        auto tNumRows = tRowMap.size() - 1;

        Kokkos::parallel_for("Vector_a * Matrix + Vector_b", Kokkos::RangePolicy<>(0, tNumRows), KOKKOS_LAMBDA(const Plato::OrdinalType & aRowOrdinal)
        {
            auto tRowStart = tRowMap(aRowOrdinal);
            auto tRowEnd = tRowMap(aRowOrdinal + 1);
            for (auto tEntryIndex = tRowStart; tEntryIndex < tRowEnd; tEntryIndex++)
            {
                auto tColumnIndex = tColIndices(tEntryIndex);
                Kokkos::atomic_add(&aOutput(tColumnIndex), tEntries(tEntryIndex) * aInput(aRowOrdinal));
            }
        });
    }
}

void
RowSummedInverseMultiply( const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                                Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo)
{
    auto tNumBlockRows = aInMatrixOne->rowMap().size()-1;
    auto tNumRows = aInMatrixOne->numRows();
    Plato::RowSum tRowSumFunctor(aInMatrixOne);
    Plato::DiagonalInverseMultiply tDiagInverseMultiplyFunctor(aInMatrixTwo);
    Plato::ScalarVector tRowSum("row sum", tNumRows);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumBlockRows), KOKKOS_LAMBDA(const Plato::OrdinalType & tBlockRowOrdinal) {
      tRowSumFunctor(tBlockRowOrdinal, tRowSum);
      tDiagInverseMultiplyFunctor(tBlockRowOrdinal, tRowSum);
    });
}

void
getDataAsNonBlock( const Teuchos::RCP<const Plato::CrsMatrixType> & aMatrix,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                         Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues,
                         int                                        aRowStride,
                         int                                        aRowOffset)
{
    const auto& tRowMap = aMatrix->rowMap();
    const auto& tColMap = aMatrix->columnIndices();
    const auto& tValues = aMatrix->entries();

    auto tNumMatrixRows = aMatrix->numRows();
    tNumMatrixRows *= aRowStride;

    auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aMatrix->numColsPerBlock();
    auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;


    // generate non block row map
    //
    aMatrixRowMap = Plato::ScalarVectorT<Plato::OrdinalType>("non block row map", tNumMatrixRows+1);

    if (aRowStride == 1)
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), KOKKOS_LAMBDA(const Plato::OrdinalType & tMatrixRowIndex) {
            auto tBlockRowIndex = tMatrixRowIndex / tNumRowsPerBlock;
            auto tLocalRowIndex = tMatrixRowIndex % tNumRowsPerBlock;
            auto tFrom = tRowMap(tBlockRowIndex);
            auto tTo   = tRowMap(tBlockRowIndex+1);
            auto tBlockRowSize = tTo - tFrom;
            aMatrixRowMap(tMatrixRowIndex) = tFrom * tBlockSize + tLocalRowIndex * tBlockRowSize * tNumColsPerBlock;
            aMatrixRowMap(tMatrixRowIndex+1) = tFrom * tBlockSize + (tLocalRowIndex+1) * tBlockRowSize * tNumColsPerBlock;
        });
    }
    else 
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), KOKKOS_LAMBDA(const Plato::OrdinalType & tMatrixRowIndex) {
            auto tBlockRowIndex = tMatrixRowIndex / aRowStride;
            auto tLocalRowIndex = tMatrixRowIndex % aRowStride;
            auto tFrom = tRowMap(tBlockRowIndex);
            auto tTo   = tRowMap(tBlockRowIndex+1);
            auto tBlockRowSize = tTo - tFrom;
            aMatrixRowMap(tMatrixRowIndex) = tFrom * aRowStride * tNumColsPerBlock + tLocalRowIndex * tBlockRowSize * tNumColsPerBlock;
            aMatrixRowMap(tMatrixRowIndex+1) = tFrom * aRowStride * tNumColsPerBlock + (tLocalRowIndex+1) * tBlockRowSize * tNumColsPerBlock;
        });
    }

    // generate non block col map and non block values
    //
    auto tNumMatrixColEntries = tColMap.extent(0)*tBlockSize*aRowStride;
    aMatrixColMap = Plato::ScalarVectorT<Plato::OrdinalType>("non block col map", tNumMatrixColEntries);
    aMatrixValues = Plato::ScalarVectorT<Plato::Scalar>     ("non block values",  tNumMatrixColEntries);

    if (aRowStride == 1)
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), KOKKOS_LAMBDA(const Plato::OrdinalType & tMatrixRowIndex) {
            auto tBlockRowIndex = tMatrixRowIndex / tNumRowsPerBlock;
            auto tLocalRowIndex = tMatrixRowIndex % tNumRowsPerBlock;
            auto tFrom = tRowMap(tBlockRowIndex);
            auto tTo   = tRowMap(tBlockRowIndex+1);
            Plato::OrdinalType tMatrixRowFrom = aMatrixRowMap(tMatrixRowIndex);
            for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
            {
                for( Plato::OrdinalType tBlockColOffset=0; tBlockColOffset<tNumColsPerBlock; ++tBlockColOffset )
                {
                    auto tMapIndex = tColMap(tColMapIndex)*tNumColsPerBlock+tBlockColOffset;
                    auto tValIndex = tColMapIndex*tBlockSize+tLocalRowIndex*tNumColsPerBlock+tBlockColOffset;
                    aMatrixColMap(tMatrixRowFrom) = tMapIndex;
                    aMatrixValues(tMatrixRowFrom) = tValues(tValIndex);
                    tMatrixRowFrom++;
                }
            }
        });
    }
    else
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), KOKKOS_LAMBDA(const Plato::OrdinalType & tMatrixRowIndex) {
            auto tBlockRowIndex = tMatrixRowIndex / aRowStride;
            auto tLocalRowIndex = tMatrixRowIndex % aRowStride;
            auto tFrom = tRowMap(tBlockRowIndex);
            auto tTo   = tRowMap(tBlockRowIndex+1);
            Plato::OrdinalType tMatrixRowFrom = aMatrixRowMap(tMatrixRowIndex);
            for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
            {
                for( Plato::OrdinalType tBlockColOffset=0; tBlockColOffset<tNumColsPerBlock; ++tBlockColOffset )
                {
                    auto tMapIndex = tColMap(tColMapIndex)*tNumColsPerBlock+tBlockColOffset;
                    auto tValIndex = tColMapIndex*tNumColsPerBlock+tBlockColOffset;
                    aMatrixColMap(tMatrixRowFrom) = tMapIndex;
                    aMatrixValues(tMatrixRowFrom) = ((tLocalRowIndex == aRowOffset) ? tValues(tValIndex) : 0.0);
                    tMatrixRowFrom++;
                }
            }
        });
    }
}

void
sortColumnEntries( const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                         Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues)
{
    auto tNumMatrixRows = aMatrixRowMap.extent(0) - 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), KOKKOS_LAMBDA(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tFrom = aMatrixRowMap(tMatrixRowIndex);
        auto tTo   = aMatrixRowMap(tMatrixRowIndex+1);
        for( auto tColMapEntryIndex_I=tFrom; tColMapEntryIndex_I<tTo; ++tColMapEntryIndex_I )
        {
            for( auto tColMapEntryIndex_J=tFrom; tColMapEntryIndex_J<tTo; ++tColMapEntryIndex_J )
            {
                if( aMatrixColMap[tColMapEntryIndex_I] < aMatrixColMap[tColMapEntryIndex_J] )
                {
                    auto tColIndex = aMatrixColMap[tColMapEntryIndex_J];
                    aMatrixColMap[tColMapEntryIndex_J] = aMatrixColMap[tColMapEntryIndex_I];
                    aMatrixColMap[tColMapEntryIndex_I] = tColIndex;
                    auto tValue = aMatrixValues[tColMapEntryIndex_J];
                    aMatrixValues[tColMapEntryIndex_J] = aMatrixValues[tColMapEntryIndex_I];
                    aMatrixValues[tColMapEntryIndex_I] = tValue;
                }
            }
        }
    });
}

void
setDataFromNonBlock(      Teuchos::RCP<Plato::CrsMatrixType>       & aMatrix,
                    const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                          Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                          Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues)
{

    sortColumnEntries(aMatrixRowMap, aMatrixColMap, aMatrixValues);

    auto tNumMatrixRows = aMatrix->numRows();
    auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aMatrix->numColsPerBlock();
    auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

    // generate block row map
    //
    auto tNumNodeRows = tNumMatrixRows/tNumRowsPerBlock;
    Plato::ScalarVectorT<Plato::OrdinalType> tRowMap("block row map", tNumNodeRows+1);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows+1), KOKKOS_LAMBDA(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tBlockRowIndex = tMatrixRowIndex / tNumRowsPerBlock;
        auto tLocalRowIndex = tMatrixRowIndex % tNumRowsPerBlock;
        if(tLocalRowIndex == 0)
          tRowMap(tBlockRowIndex) = aMatrixRowMap(tMatrixRowIndex)/tBlockSize;
    });
    aMatrix->setRowMap(tRowMap);

    // generate block col map and block values
    //
    auto tNumBlockMatEntries = aMatrixValues.extent(0);
    auto tNumBlockColEntries = tNumBlockMatEntries / tBlockSize;
    Plato::ScalarVectorT<Plato::OrdinalType> tColMap("block col map", tNumBlockColEntries);
    Plato::ScalarVectorT<Plato::Scalar>      tValues("block values",  tNumBlockMatEntries);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), KOKKOS_LAMBDA(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tBlockRowIndex = tMatrixRowIndex / tNumRowsPerBlock;
        auto tLocalRowIndex = tMatrixRowIndex % tNumRowsPerBlock;
        auto tFrom = tRowMap(tBlockRowIndex);
        auto tTo   = tRowMap(tBlockRowIndex+1);
        Plato::OrdinalType tMatrixRowFrom = aMatrixRowMap(tMatrixRowIndex);
        for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
        {
            if(tLocalRowIndex == 0)
            {
                tColMap(tColMapIndex) = aMatrixColMap(tMatrixRowFrom)/tNumColsPerBlock;
            }
            for( Plato::OrdinalType tBlockColOffset=0; tBlockColOffset<tNumColsPerBlock; ++tBlockColOffset )
            {
                auto tValIndex = tColMapIndex*tBlockSize + tLocalRowIndex*tNumColsPerBlock + tBlockColOffset;
                tValues(tValIndex) = aMatrixValues(tMatrixRowFrom++);
            }
        }
    });
    aMatrix->setColumnIndices(tColMap);
    aMatrix->setEntries(tValues);
}

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

void
MatrixMatrixMultiply( const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
                            Teuchos::RCP<Plato::CrsMatrixType> & aOutMatrix,
                            SPGEMMAlgorithm                      aAlgorithm)
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    typedef Plato::ScalarVectorT<OrdinalType> OrdinalView;
    typedef Plato::ScalarVectorT<Scalar>  ScalarView;

    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <OrdinalType, OrdinalType, Scalar,
        typename Plato::ExecSpace, 
        typename Plato::MemSpace,
        typename Plato::MemSpace > KernelHandle;

    KernelHandle tKernel;
    tKernel.set_team_work_size(1);
    tKernel.set_dynamic_scheduling(false);

    tKernel.create_spgemm_handle(aAlgorithm);

    const auto& tMatOne = *aInMatrixOne;
    const auto& tMatTwo = *aInMatrixTwo;
    auto& tOutMat = *aOutMatrix;
    const OrdinalType tNumRowsOne = tMatOne.numRows();
    const OrdinalType tNumColsOne = tMatOne.numCols();
    const OrdinalType tNumRowsTwo = tMatTwo.numRows();
    const OrdinalType tNumColsTwo = tMatTwo.numCols();
    const OrdinalType tNumRowsOut = tOutMat.numRows();
    const OrdinalType tNumColsOut = tOutMat.numCols();

    // C = M1 x M2
    //
    // numCols(M1) === numRows(M2)
    if (tNumRowsTwo != tNumColsOne) { ANALYZE_THROWERR("input matrices have incompatible shapes"); }

    // numRows(C)  === numRows(M1)
    if (tNumRowsOut != tNumRowsOne) { ANALYZE_THROWERR("output matrix has incorrect shape"); }

    // numCols(C)  === numCols(M2)
    if (tNumColsOut != tNumColsTwo) { ANALYZE_THROWERR("output matrix has incorrect shape"); }

    // Get matrix data in non-block form
    ScalarView tMatOneValues;
    OrdinalView tMatOneRowMap, tMatOneColMap;
    if (aInMatrixOne->isBlockMatrix())
    {
      Plato::getDataAsNonBlock(aInMatrixOne, tMatOneRowMap, tMatOneColMap, tMatOneValues);
    }
    else
    {
      tMatOneValues = tMatOne.entries();
      tMatOneRowMap = tMatOne.rowMap();
      tMatOneColMap = tMatOne.columnIndices();
    }

    ScalarView tMatTwoValues;
    OrdinalView tMatTwoRowMap, tMatTwoColMap;
    if (aInMatrixTwo->isBlockMatrix())
    {
      Plato::getDataAsNonBlock(aInMatrixTwo, tMatTwoRowMap, tMatTwoColMap, tMatTwoValues);
    }
    else
    {
      tMatTwoValues = tMatTwo.entries();
      tMatTwoRowMap = tMatTwo.rowMap();
      tMatTwoColMap = tMatTwo.columnIndices();
    }

    KokkosSparse::sort_crs_matrix<Plato::ExecSpace>(tMatOneRowMap, tMatOneColMap, tMatOneValues);
    KokkosSparse::sort_crs_matrix<Plato::ExecSpace>(tMatTwoRowMap, tMatTwoColMap, tMatTwoValues);

    constexpr bool transpose = false;
    OrdinalView tOutRowMap ("output row map", tNumRowsOne + 1);
    spgemm_symbolic ( &tKernel, tNumRowsOne, tNumRowsTwo, tNumColsTwo,
        tMatOneRowMap, tMatOneColMap, transpose,
        tMatTwoRowMap, tMatTwoColMap, transpose,
        tOutRowMap
    );

    OrdinalView tOutColMap;
    ScalarView  tOutValues;
    size_t tNumOutValues = tKernel.get_spgemm_handle()->get_c_nnz();
    if (tNumOutValues){
      tOutColMap = OrdinalView(Kokkos::ViewAllocateWithoutInitializing("out column map"), tNumOutValues);
      tOutValues = ScalarView (Kokkos::ViewAllocateWithoutInitializing("out values"),  tNumOutValues);
    }
    spgemm_numeric( &tKernel, tNumRowsOne, tNumRowsTwo, tNumColsTwo,
        tMatOneRowMap, tMatOneColMap, tMatOneValues, /*transpose=*/false,
        tMatTwoRowMap, tMatTwoColMap, tMatTwoValues, /*transpose=*/false,
        tOutRowMap, tOutColMap, tOutValues
    );

    // update out matrix
    if (aOutMatrix->isBlockMatrix())
    {
      Plato::setDataFromNonBlock(aOutMatrix, tOutRowMap, tOutColMap, tOutValues);
    }
    else
    {
      aOutMatrix->setRowMap(tOutRowMap);
      aOutMatrix->setColumnIndices(tOutColMap);
      aOutMatrix->setEntries(tOutValues);
    }

    tKernel.destroy_spgemm_handle();
}

void
MatrixMinusMatrix(      Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                  const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
                        Plato::OrdinalType aOffset)
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    typedef Plato::ScalarVectorT<OrdinalType> OrdinalView;
    typedef Plato::ScalarVectorT<Scalar>  ScalarView;

    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <OrdinalType, OrdinalType, Scalar,
        typename Plato::ExecSpace, 
        typename Plato::MemSpace,
        typename Plato::MemSpace > KernelHandle;

    const auto& tMatOne = *aInMatrixOne;
    const auto& tMatTwo = *aInMatrixTwo;
    const OrdinalType tNumRowsOne = tMatOne.numRows();
    const OrdinalType tNumColsOne = tMatOne.numCols();
    const OrdinalType tNumRowsTwo = tMatTwo.numRows();
    const OrdinalType tNumColsTwo = tMatTwo.numCols();

    auto tNumRowsPerBlock = tMatOne.numRowsPerBlock();

    if (tNumColsOne != tNumColsTwo) { ANALYZE_THROWERR("matrices have incompatible shape"); }

    ScalarView tMatOneValues;
    OrdinalView tMatOneRowMap, tMatOneColMap;
    Plato::getDataAsNonBlock(aInMatrixOne, tMatOneRowMap, tMatOneColMap, tMatOneValues);

    ScalarView tMatTwoValues;
    OrdinalView tMatTwoRowMap, tMatTwoColMap;
    Plato::getDataAsNonBlock(aInMatrixTwo, tMatTwoRowMap, tMatTwoColMap, tMatTwoValues, tNumRowsPerBlock, aOffset);

    OrdinalView tOutRowMap ("output row map", tNumRowsOne + 1);

    KernelHandle tKernel;
    tKernel.create_spadd_handle(/*sort rows=*/ false);
    auto tAddHandle = tKernel.get_spadd_handle();
    KokkosSparse::Experimental::spadd_symbolic< KernelHandle,
      OrdinalView, OrdinalView,
      OrdinalView, OrdinalView,
      OrdinalView
    >
    ( &tKernel,
      tMatOneRowMap, tMatOneColMap,
      tMatTwoRowMap, tMatTwoColMap,
      tOutRowMap
    );

    auto t_nnz = tAddHandle->get_c_nnz();

    OrdinalView tOutColMap("output graph", t_nnz);
    ScalarView  tOutValues("output values", t_nnz);
    KokkosSparse::Experimental::spadd_numeric< KernelHandle,
      OrdinalView, OrdinalView, Scalar, ScalarView,
      OrdinalView, OrdinalView, Scalar, ScalarView,
      OrdinalView, OrdinalView, ScalarView
    >
    ( &tKernel,
      tMatOneRowMap, tMatOneColMap, tMatOneValues, 1.0,
      tMatTwoRowMap, tMatTwoColMap, tMatTwoValues, -1.0,
      tOutRowMap,    tOutColMap,    tOutValues
    );

    Plato::setDataFromNonBlock(aInMatrixOne, tOutRowMap, tOutColMap, tOutValues);
    tKernel.destroy_spadd_handle();
}

void Condense(       Teuchos::RCP<Plato::CrsMatrixType> & aA,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aB,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aC,
                            Teuchos::RCP<Plato::CrsMatrixType> & aD,
                            Plato::OrdinalType aOffset )
{
  RowSummedInverseMultiply ( aC, aD );

  auto tNumRows = aB->numRows();
  auto tNumCols = aD->numCols();
  auto tNumRowsPerBlock = aB->numRowsPerBlock();
  auto tNumColsPerBlock = aD->numColsPerBlock();
  auto tBD = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock ) );
  MatrixMatrixMultiply     ( aB, aD, tBD );

  MatrixMinusMatrix        ( aA, tBD, aOffset );
}

void
MatrixTranspose( const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                       Teuchos::RCP<Plato::CrsMatrixType> & aMatrixTranspose)
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    typedef Plato::ScalarVectorT<OrdinalType> OrdinalView;
    typedef Plato::ScalarVectorT<Scalar>  ScalarView;

    ScalarView tEntries;
    OrdinalView tRowMap, tColMap;
    if (aMatrix->isBlockMatrix())
    {
      Plato::getDataAsNonBlock(aMatrix, tRowMap, tColMap, tEntries);
    }
    else
    {
      tEntries = aMatrix->entries();
      tRowMap = aMatrix->rowMap();
      tColMap = aMatrix->columnIndices();
    }

    OrdinalType tNumRowsT = aMatrix->numCols();
    OrdinalView tRowMapT("row map", tNumRowsT+1);
    OrdinalView tColMapT("col map", tEntries.size());
    ScalarView tEntriesT("entries", tEntries.size());

    // determine rowmap
    OrdinalType tNumRows = tRowMap.size() - 1;
    Kokkos::parallel_for("nonzeros", Kokkos::RangePolicy<OrdinalType>(0, tNumRows), KOKKOS_LAMBDA(OrdinalType iRowOrdinal)
    {
        auto tRowStart = tRowMap(iRowOrdinal);
        auto tRowEnd = tRowMap(iRowOrdinal + 1);
        for (auto tEntryIndex = tRowStart; tEntryIndex < tRowEnd; tEntryIndex++)
        {
            auto iColumnIndex = tColMap(tEntryIndex);
            Kokkos::atomic_increment(&tRowMapT(iColumnIndex));
        }
    });

    OrdinalType tNumEntries(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalType>(0,tNumRowsT+1),
    KOKKOS_LAMBDA (const OrdinalType& iOrdinal, OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const OrdinalType tVal = tRowMapT(iOrdinal);
        if( tIsFinal )
        {
          tRowMapT(iOrdinal) = aUpdate;
        }
        aUpdate += tVal;
    }, tNumEntries);

    // determine column map and entries
    OrdinalView tOffsetT("offsets", tNumRowsT);
    Kokkos::parallel_for("colmap and entries", Kokkos::RangePolicy<OrdinalType>(0, tNumRows), KOKKOS_LAMBDA(OrdinalType iRowOrdinal)
    {
        auto tRowStart = tRowMap(iRowOrdinal);
        auto tRowEnd = tRowMap(iRowOrdinal + 1);
        for (auto iEntryIndex = tRowStart; iEntryIndex < tRowEnd; iEntryIndex++)
        {
            auto iRowIndexT = tColMap(iEntryIndex);
            auto tMyOffset = Kokkos::atomic_fetch_add(&tOffsetT(iRowIndexT), 1);
            auto iEntryIndexT = tRowMapT(iRowIndexT)+tMyOffset;
            tColMapT(iEntryIndexT) = iRowOrdinal;
            tEntriesT(iEntryIndexT) = tEntries(iEntryIndex);
        }
    });

    // update matrix transpose 
    if (aMatrixTranspose->isBlockMatrix())
    {
      Plato::setDataFromNonBlock(aMatrixTranspose, tRowMapT, tColMapT, tEntriesT);
    }
    else
    {
      aMatrixTranspose->setRowMap(tRowMapT);
      aMatrixTranspose->setColumnIndices(tColMapT);
      aMatrixTranspose->setEntries(tEntriesT);
    }
}

} 
