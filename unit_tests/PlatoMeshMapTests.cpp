#include "util/PlatoTestHelpers.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <sstream>
#include <fstream>
#include <stdio.h>

#include "Tet4.hpp"
#include "ElementBase.hpp"

#include "Plato_InputData.hpp"
#include "Plato_Exceptions.hpp"
#include "Plato_Parser.hpp"

#define MAKE_PUBLIC
#include "Plato_MeshMap.hpp"

namespace PlatoTestMeshMap {

namespace {
namespace pth = Plato::TestHelpers;
using SparseMatrix = Plato::Geometry::MeshMap<Plato::Scalar>::SparseMatrix;

/***************************************************************************//**
* \brief Convert sparse matrix to full matrix
*******************************************************************************/
std::vector<std::vector<Plato::Scalar>> to_full(const SparseMatrix &aInMatrix) {
  using OrdinalType = Plato::Geometry::MeshMap<Plato::Scalar>::OrdinalT;
  using Plato::Scalar;

  std::vector<std::vector<Scalar>> retMatrix(
      aInMatrix.mNumRows, std::vector<Scalar>(aInMatrix.mNumCols, 0.0));

  const auto tRowMap = pth::get(aInMatrix.mRowMap);
  const auto tColMap = pth::get(aInMatrix.mColMap);
  const auto tValues = pth::get(aInMatrix.mEntries);

  const auto tNumRows = aInMatrix.mNumRows;
  for (OrdinalType iRowIndex = 0; iRowIndex < tNumRows; iRowIndex++) {
    const auto tFrom = tRowMap(iRowIndex);
    const auto tTo = tRowMap(iRowIndex + 1);
    for (auto iEntryIndex = tFrom; iEntryIndex < tTo; iEntryIndex++) {
      const auto iColIndex = tColMap(iEntryIndex);
      retMatrix[iRowIndex][iColIndex] = tValues(iEntryIndex);
    }
  }
  return retMatrix;
}

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
} // namespace

/******************************************************************************/
/*!
  \brief Compute basis function values of Tet4

  1. Compute the centroid of each element in a test mesh in physical coordinates.
  2. Use GetBases() to determine the basis function values at the equivalent
     point in element coordinates.

     X_i = N(x)_I v^e_{Ii}

     v^e_{Ii}: vertex coordinates for local node I of element e
     X_i:      Input point for which basis values are determined. Element
               centroid in this test.
     x:        location of X in element coordinates.
     N(x)_I:   Basis values to be determines.

  test passes if:
    N(x)_I = {1/4, 1/4, 1/4, 1/4};
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, GetBasis_Tet4)
  {

    // create mesh
    //
    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);

    // create GetBasis functor
    //
    using ElementType = Plato::Tet4;
    Plato::Geometry::GetBasis<ElementType, Plato::Scalar> tGetBasis(tMesh);

    auto tNElems = tMesh->NumElements();
    Plato::ScalarMultiVector tBases("basis values", ElementType::mNumNodesPerCell, tNElems);

    auto tCoords = tMesh->Coordinates();
    auto tCells2Nodes = tMesh->Connectivity();

    // map from input to output
    //
    Kokkos::parallel_for("compute", Kokkos::RangePolicy<int>(0, tNElems), KOKKOS_LAMBDA(int aOrdinal)
    {
        Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tElemBases(0.0);
        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);

        // compute element centroid
        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            auto iVertOrdinal = tCells2Nodes(aOrdinal*ElementType::mNumNodesPerCell+iVert);
            for(int iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
            {
                tInPoint(iDim) += tCoords(iVertOrdinal*ElementType::mNumSpatialDims+iDim)
                                / ElementType::mNumNodesPerCell;
            }
        }

        tGetBasis(aOrdinal, tInPoint, tElemBases);

        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            tBases(iVert, aOrdinal) = tElemBases(iVert);
        }
    });

    double tol_double = 1e-14;
    auto tBases_host = pth::get(tBases);
    for(int iElem=0; iElem<tNElems; iElem++)
    {
        for(int iNode=0; iNode<(ElementType::mNumNodesPerCell); iNode++)
        {
            TEST_FLOATING_EQUALITY(tBases_host(iNode, iElem), 1.0/4.0, tol_double);
        }
    }
  }

/******************************************************************************/
/*!
  \brief Compute basis function values of Tet10

  1. Compute the centroid of each element in a test mesh in physical coordinates.
  2. Use GetBases() to determine the basis function values at the equivalent
     point in element coordinates.

     X_i = N(x)_I v^e_{Ii}

     v^e_{Ii}: vertex coordinates for local node I of element e
     X_i:      Input point for which basis values are determined. Element
               centroid in this test.
     x:        location of X in element coordinates.
     N(x)_I:   Basis values to be determines.

  test passes if:
    N(x)_I = {-1/8, -1/8, -1/8, -1/8, 1/4, 1/4, 1/4, 1/4, 1/4, 1/4};
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, GetBasis_Tet10)
  {

    // create mesh
    //
    constexpr int cMeshWidth=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", cMeshWidth);

    // create GetBasis functor
    //
    using ElementType = Plato::Tet10;
    Plato::Geometry::GetBasis<ElementType, Plato::Scalar> tGetBasis(tMesh);

    auto tNElems = tMesh->NumElements();
    Plato::ScalarMultiVector tBases("basis values", ElementType::mNumNodesPerCell, tNElems);

    auto tCoords = tMesh->Coordinates();
    auto tCells2Nodes = tMesh->Connectivity();

    // map from input to output
    //
    Kokkos::parallel_for("compute", Kokkos::RangePolicy<int>(0, tNElems), KOKKOS_LAMBDA(int aOrdinal)
    {
        Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tElemBases(0.0);
        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);

        // compute element centroid
        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            auto iVertOrdinal = tCells2Nodes(aOrdinal*ElementType::mNumNodesPerCell+iVert);
            for(int iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
            {
                tInPoint(iDim) += tCoords(iVertOrdinal*ElementType::mNumSpatialDims+iDim)
                                / ElementType::mNumNodesPerCell;
            }
        }

        tGetBasis(aOrdinal, tInPoint, tElemBases);

        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            tBases(iVert, aOrdinal) = tElemBases(iVert);
        }
    });

    auto oe = Plato::Scalar(1)/8;
    auto of = Plato::Scalar(1)/4;
    std::vector<Plato::Scalar> tBases_gold = {-oe, -oe, -oe, -oe, of, of, of, of, of, of};

    double tol_double = 1e-14;
    auto tBases_host = pth::get(tBases);
    for(int iElem=0; iElem<tNElems; iElem++)
    {
        for(int iNode=0; iNode<(ElementType::mNumNodesPerCell); iNode++)
        {
            TEST_FLOATING_EQUALITY(tBases_host(iNode, iElem), tBases_gold[iNode], tol_double);
        }
    }
  }

/******************************************************************************/
/*!
  \brief Compute basis function values of Hex8

  1. Compute the point halfway between the centroid and node0 of each element
     in a test mesh in physical coordinates.
  2. Use GetBases() to determine the basis function values at the equivalent
     point in element coordinates.

     X_i = N(x)_I v^e_{Ii}

     v^e_{Ii}: vertex coordinates for local node I of element e
     X_i:      Input point for which basis values are determined. Element
               centroid in this test.
     x:        location of X in element coordinates.
     N(x)_I:   Basis values to be determines.

  test passes if:
    N(x)_I = {27/64, 9/64, 3/64, 9/64, 9/64, 3/64, 1/64, 3/64};
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, GetBasis_Hex8)
  {

    // create mesh
    //
    constexpr int cMeshWidth=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("HEX8", cMeshWidth);

    // create GetBasis functor
    //
    using ElementType = Plato::Hex8;
    Plato::Geometry::GetBasis<ElementType, Plato::Scalar> tGetBasis(tMesh);

    auto tNElems = tMesh->NumElements();
    Plato::ScalarMultiVector tBases("basis values", ElementType::mNumNodesPerCell, tNElems);

    auto tCoords = tMesh->Coordinates();
    auto tCells2Nodes = tMesh->Connectivity();

    // map from input to output
    //
    Kokkos::parallel_for("compute", Kokkos::RangePolicy<int>(0, tNElems), KOKKOS_LAMBDA(int aOrdinal)
    {
        Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tElemBases(0.0);
        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);

        // compute the point halfway between the element centroid and node 0 of the element
        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            auto iVertOrdinal = tCells2Nodes(aOrdinal*ElementType::mNumNodesPerCell+iVert);
            for(int iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
            {
                tInPoint(iDim) += tCoords(iVertOrdinal*ElementType::mNumSpatialDims+iDim)
                                / ElementType::mNumNodesPerCell;
            }
        }
        int iVert = 0;
        auto iVertOrdinal = tCells2Nodes(aOrdinal*ElementType::mNumNodesPerCell+iVert);
        for(int iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
        {
            tInPoint(iDim) += tCoords(iVertOrdinal*ElementType::mNumSpatialDims+iDim);
            tInPoint(iDim) /= 2;
        }

        tGetBasis(aOrdinal, tInPoint, tElemBases);

        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            tBases(iVert, aOrdinal) = tElemBases(iVert);
        }
    });

    std::vector<Plato::Scalar> tBases_gold = {
        Plato::Scalar(27)/64, Plato::Scalar(9)/64, Plato::Scalar(3)/64, Plato::Scalar(9)/64,
        Plato::Scalar(9)/64, Plato::Scalar(3)/64, Plato::Scalar(1)/64, Plato::Scalar(3)/64
    };

    double tol_double = 1e-14;
    auto tBases_host = pth::get(tBases);
    for(int iElem=0; iElem<tNElems; iElem++)
    {
        for(int iNode=0; iNode<(ElementType::mNumNodesPerCell); iNode++)
        {
            TEST_FLOATING_EQUALITY(tBases_host(iNode, iElem), tBases_gold[iNode], tol_double);
        }
    }
  }

/******************************************************************************/
/*!
  \brief Compute basis function values of Quad4

  1. Compute the point halfway between the centroid and node0 of each element
     in a test mesh in physical coordinates.
  2. Use GetBases() to determine the basis function values at the equivalent
     point in element coordinates.

     X_i = N(x)_I v^e_{Ii}

     v^e_{Ii}: vertex coordinates for local node I of element e
     X_i:      Input point for which basis values are determined. Element
               centroid in this test.
     x:        location of X in element coordinates.
     N(x)_I:   Basis values to be determines.

  test passes if:
    N(x)_I = {27/64, 9/64, 3/64, 9/64, 9/64, 3/64, 1/64, 3/64};
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, GetBasis_Quad4)
  {

    // create mesh
    //
    constexpr int cMeshWidth=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("QUAD4", cMeshWidth);

    // create GetBasis functor
    //
    using ElementType = Plato::Quad4;
    Plato::Geometry::GetBasis<ElementType, Plato::Scalar> tGetBasis(tMesh);

    auto tNElems = tMesh->NumElements();
    Plato::ScalarMultiVector tBases("basis values", ElementType::mNumNodesPerCell, tNElems);

    auto tCoords = tMesh->Coordinates();
    auto tCells2Nodes = tMesh->Connectivity();

    // map from input to output
    //
    Kokkos::parallel_for("compute", Kokkos::RangePolicy<int>(0, tNElems), KOKKOS_LAMBDA(int aOrdinal)
    {
        Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tElemBases(0.0);
        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);

        // compute the point halfway between the element centroid and node 0 of the element
        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            auto iVertOrdinal = tCells2Nodes(aOrdinal*ElementType::mNumNodesPerCell+iVert);
            for(int iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
            {
                tInPoint(iDim) += tCoords(iVertOrdinal*ElementType::mNumSpatialDims+iDim)
                                / ElementType::mNumNodesPerCell;
            }
        }
        int iVert = 0;
        auto iVertOrdinal = tCells2Nodes(aOrdinal*ElementType::mNumNodesPerCell+iVert);
        for(int iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
        {
            tInPoint(iDim) += tCoords(iVertOrdinal*ElementType::mNumSpatialDims+iDim);
            tInPoint(iDim) /= 2;
        }

        tGetBasis(aOrdinal, tInPoint, tElemBases);

        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            tBases(iVert, aOrdinal) = tElemBases(iVert);
        }
    });

    std::vector<Plato::Scalar> tBases_gold = {
        Plato::Scalar(9)/16, Plato::Scalar(3)/16, Plato::Scalar(1)/16, Plato::Scalar(3)/16
    };

    double tol_double = 1e-14;
    auto tBases_host = pth::get(tBases);
    for(int iElem=0; iElem<tNElems; iElem++)
    {
        for(int iNode=0; iNode<(ElementType::mNumNodesPerCell); iNode++)
        {
            TEST_FLOATING_EQUALITY(tBases_host(iNode, iElem), tBases_gold[iNode], tol_double);
        }
    }
  }

/******************************************************************************/
/*!
  \brief Compute basis function values of Hex27

  1. Compute the point halfway between the centroid and node0 of each element
     in a test mesh in physical coordinates.
  2. Use GetBases() to determine the basis function values at the equivalent
     point in element coordinates.

     X_i = N(x)_I v^e_{Ii}

     v^e_{Ii}: vertex coordinates for local node I of element e
     X_i:      Input point for which basis values are determined. Element
               centroid in this test.
     x:        location of X in element coordinates.
     N(x)_I:   Basis values to be determines.

  test passes if:
    N(x)_I = {27/512, -9/512,  3/512, -9/512, -9/512,  3/512, -1/512,  3/512, 27/256,
              -9/256, -9/256, 27/256, 27/256, -9/256,  3/256, -9/256, -9/256,  3/256,
               3/256, -9/256, 27/64,  27/128, -9/128, 27/128, -9/128, 27/128, -9/128};
              
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, GetBasis_Hex27)
  {

    // create mesh
    //
    constexpr int cMeshWidth=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("HEX27", cMeshWidth);

    // create GetBasis functor
    //
    using ElementType = Plato::Hex27;
    Plato::Geometry::GetBasis<ElementType, Plato::Scalar> tGetBasis(tMesh);

    auto tNElems = tMesh->NumElements();
    Plato::ScalarMultiVector tBases("basis values", ElementType::mNumNodesPerCell, tNElems);

    auto tCoords = tMesh->Coordinates();
    auto tCells2Nodes = tMesh->Connectivity();

    // map from input to output
    //
    Kokkos::parallel_for("compute", Kokkos::RangePolicy<int>(0, tNElems), KOKKOS_LAMBDA(int aOrdinal)
    {
        Plato::Array<ElementType::mNumNodesPerCell, Plato::Scalar> tElemBases(0.0);
        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInPoint(0.0);

        // compute the point halfway between the element centroid and node 0 of the element
        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            auto iVertOrdinal = tCells2Nodes(aOrdinal*ElementType::mNumNodesPerCell+iVert);
            for(int iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
            {
                tInPoint(iDim) += tCoords(iVertOrdinal*ElementType::mNumSpatialDims+iDim)
                                / ElementType::mNumNodesPerCell;
            }
        }
        int iVert = 0;
        auto iVertOrdinal = tCells2Nodes(aOrdinal*ElementType::mNumNodesPerCell+iVert);
        for(int iDim=0; iDim<ElementType::mNumSpatialDims; iDim++)
        {
            tInPoint(iDim) += tCoords(iVertOrdinal*ElementType::mNumSpatialDims+iDim);
            tInPoint(iDim) /= 2;
        }

        tGetBasis(aOrdinal, tInPoint, tElemBases);

        for(int iVert=0; iVert<(ElementType::mNumNodesPerCell); iVert++)
        {
            tBases(iVert, aOrdinal) = tElemBases(iVert);
        }
    });

    std::vector<Plato::Scalar> tBases_gold = {
      Plato::Scalar(27)/512, Plato::Scalar(-9)/512, Plato::Scalar( 3)/512,
      Plato::Scalar(-9)/512, Plato::Scalar(-9)/512, Plato::Scalar( 3)/512,
      Plato::Scalar(-1)/512, Plato::Scalar( 3)/512, Plato::Scalar(27)/256,
      Plato::Scalar(-9)/256, Plato::Scalar(-9)/256, Plato::Scalar(27)/256,
      Plato::Scalar(27)/256, Plato::Scalar(-9)/256, Plato::Scalar( 3)/256,
      Plato::Scalar(-9)/256, Plato::Scalar(-9)/256, Plato::Scalar( 3)/256,
      Plato::Scalar( 3)/256, Plato::Scalar(-9)/256, Plato::Scalar(27)/64,
      Plato::Scalar(27)/128, Plato::Scalar(-9)/128, Plato::Scalar(27)/128,
      Plato::Scalar(-9)/128, Plato::Scalar(27)/128, Plato::Scalar(-9)/128
    };

    double tol_double = 1e-14;
    auto tBases_host = pth::get(tBases);
    for(int iElem=0; iElem<tNElems; iElem++)
    {
        for(int iNode=0; iNode<(ElementType::mNumNodesPerCell); iNode++)
        {
            TEST_FLOATING_EQUALITY(tBases_host(iNode, iElem), tBases_gold[iNode], tol_double);
        }
    }
  }


/******************************************************************************/
/*!
  \brief Test symmetry plane implementation.

  test passes if points are mirrored correctly
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryPlane)
  {

    // create input for SymmetryPlane
    //
    double rx = 0.1, ry = 0.2, rz = 0.3;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<LinearMap>";
    input << "  <Type>SymmetryPlane</Type>";
    input << "  <Origin>";
    input << "    <X>" << rx << "</X>";
    input << "    <Y>" << ry << "</Y>";
    input << "    <Z>" << rz << "</Z>";
    input << "  </Origin>";
    input << "  <Normal>";
    input << "    <X>" << nx << "</X>";
    input << "    <Y>" << ny << "</Y>";
    input << "    <Z>" << nz << "</Z>";
    input << "  </Normal>";
    input << "</LinearMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create SymmetryPlane from input
    //
    constexpr int tNumDims = 3;
    auto tMathMapParams = tInputData.get<Plato::InputData>("LinearMap");
    Plato::Geometry::SymmetryPlane<tNumDims, Plato::Scalar> tMathMap(tMathMapParams);

    // create input and output views
    //
    int tNumVals = 2;
    Plato::ScalarMultiVector tXin("Xin", tNumDims, tNumVals);
    Plato::ScalarMultiVector tXout("Xin", tNumDims, tNumVals);
    auto tXin_host = Kokkos::create_mirror_view(tXin);

    double p0_X = 0.0, p0_Y = 0.0, p0_Z = 0.0;
    double p1_X = 0.0, p1_Y = 0.0, p1_Z = 0.5;

    tXin_host(0,0) = p0_X; tXin_host(1,0) = p0_Y; tXin_host(2,0) = p0_Z;
    tXin_host(0,1) = p1_X; tXin_host(1,1) = p1_Y; tXin_host(2,1) = p1_Z;
    Kokkos::deep_copy(tXin, tXin_host);

    // map from input to output
    //
    Kokkos::parallel_for("compute", Kokkos::RangePolicy<int>(0, tNumVals), KOKKOS_LAMBDA(int aOrdinal)
    {
        tMathMap(aOrdinal, tXin, tXout);
    });

    // test results
    //
    auto tXout_host = Kokkos::create_mirror_view(tXout);
    Kokkos::deep_copy(tXout_host, tXout);

    double tol_double = 1e-14;
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_X, /*Result=*/ tXout_host(0,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_Y, /*Result=*/ tXout_host(1,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_Z-2.0*(p0_Z-rz)*nz, /*Result=*/ tXout_host(2,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_X, /*Result=*/ tXout_host(0,1), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_Y, /*Result=*/ tXout_host(1,1), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_Z, /*Result=*/ tXout_host(2,1), tol_double);
  }

/******************************************************************************/
/*!
  \brief Test translation implementation.

  test passes if points are translated correctly
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, Translation)
  {

    // create input for Translation
    //
    double tx = 0.0, ty = 1.0, tz = 0.0;

    std::stringstream input;
    input << "<LinearMap>";
    input << "  <Type>Translation</Type>";
    input << "  <Vector>";
    input << "    <X>" << tx << "</X>";
    input << "    <Y>" << ty << "</Y>";
    input << "    <Z>" << tz << "</Z>";
    input << "  </Vector>";
    input << "</LinearMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create Translation from input
    //
    constexpr int tNumDims = 3;
    auto tMathMapParams = tInputData.get<Plato::InputData>("LinearMap");
    Plato::Geometry::Translation<tNumDims, Plato::Scalar> tMathMap(tMathMapParams);

    // create input and output views
    //
    int tNumVals = 2;
    Plato::ScalarMultiVector tXin("Xin", tNumDims, tNumVals);
    Plato::ScalarMultiVector tXout("Xin", tNumDims, tNumVals);
    auto tXin_host = Kokkos::create_mirror_view(tXin);

    double p0_X = 0.3, p0_Y = 0.0, p0_Z = 1.0;
    double p1_X = 0.8, p1_Y = 0.5, p1_Z = 5.9;

    tXin_host(0,0) = p0_X; tXin_host(1,0) = p0_Y; tXin_host(2,0) = p0_Z;
    tXin_host(0,1) = p1_X; tXin_host(1,1) = p1_Y; tXin_host(2,1) = p1_Z;

    Kokkos::deep_copy(tXin, tXin_host);

    // map from input to output
    //
    Kokkos::parallel_for("compute", Kokkos::RangePolicy<int>(0, tNumVals), KOKKOS_LAMBDA(int aOrdinal)
    {
        tMathMap(aOrdinal, tXin, tXout);
    });

    // test results
    //
    auto tXout_host = Kokkos::create_mirror_view(tXout);
    Kokkos::deep_copy(tXout_host, tXout);

    double tol_double = 1e-14;
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_X+tx, /*Result=*/ tXout_host(0,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_Y+ty, /*Result=*/ tXout_host(1,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_Z+tz, /*Result=*/ tXout_host(2,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_X+tx, /*Result=*/ tXout_host(0,1), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_Y+ty, /*Result=*/ tXout_host(1,1), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_Z+tz, /*Result=*/ tXout_host(2,1), tol_double);
  }

/******************************************************************************/
/*!
  \brief Enforce symmetry on a linear field on an asymmetric tet mesh.

  The linear tet mesh, while asymmetric, can approximate the symmetrized linear
  field accurately.  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  then constructs a field p(z) = z in {0.0,1.0} and applies the MeshMap, f(p)

  test passes if:
    f(p(z)) == z   for z > 0.5
    f(p(z)) == 1-z for z < 0.5
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryMeshMap_Tet10)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tCoords = tMesh->Coordinates();
    auto tNVerts = tMesh->NumNodes();

    auto tDim = tMesh->NumDimensions();
    Kokkos::View<double*, MemSpace> tInField("not symmetric", tNVerts);
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    Kokkos::parallel_for("compute field", Kokkos::RangePolicy<OrdinalType>(0, tNVerts), KOKKOS_LAMBDA(OrdinalType iVertOrdinal)
    {
        tInField(iVertOrdinal) = tCoords(iVertOrdinal*tDim+2);
    });

    Kokkos::View<double*, MemSpace> tOutField("symmetric", tNVerts);
    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    for(OrdinalType i=0; i<tNVerts; i++)
    {
        if(tInField_host(i) > 1e-15)
        {
            if(tInField_host(i) < rz )
            {
                TEST_FLOATING_EQUALITY(2*rz-tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) > rz )
            {
                TEST_FLOATING_EQUALITY(tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) == 0.0 )
            {
                TEST_ASSERT(tOutField_host(i) == 0.0);
            }
        }
    }
  }
/******************************************************************************/
/*!
  \brief Enforce symmetry on a linear field on an asymmetric tet mesh.

  The linear tet mesh, while asymmetric, can approximate the symmetrized linear
  field accurately.  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  then constructs a field p(z) = z in {0.0,1.0} and applies the MeshMap, f(p)

  test passes if:
    f(p(z)) == z   for z > 0.5
    f(p(z)) == 1-z for z < 0.5
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryMeshMap)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tCoords = tMesh->Coordinates();
    auto tNVerts = tMesh->NumNodes();

    auto tDim = tMesh->NumDimensions();
    Kokkos::View<double*, MemSpace> tInField("not symmetric", tNVerts);
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    Kokkos::parallel_for("compute field", Kokkos::RangePolicy<OrdinalType>(0, tNVerts), KOKKOS_LAMBDA(OrdinalType iVertOrdinal)
    {
        tInField(iVertOrdinal) = tCoords(iVertOrdinal*tDim+2);
    });

    Kokkos::View<double*, MemSpace> tOutField("symmetric", tNVerts);
    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    for(OrdinalType i=0; i<tNVerts; i++)
    {
        if(tInField_host(i) > 1e-15)
        {
            if(tInField_host(i) < rz )
            {
                TEST_FLOATING_EQUALITY(2*rz-tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) > rz )
            {
                TEST_FLOATING_EQUALITY(tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) == 0.0 )
            {
                TEST_ASSERT(tOutField_host(i) == 0.0);
            }
        }
    }
  }

/******************************************************************************/
/*!
  \brief Enforce symmetry on a linear field on an asymmetric tet mesh.

  The linear tet mesh, while asymmetric, can approximate the symmetrized linear
  field accurately.  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  then constructs a field p(z) = z in {0.0,1.0} and applies the MeshMap, f(p)

  test passes if:
    f(p(z)) == z   for z > 0.5
    f(p(z)) == 1-z for z < 0.5
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryMeshMap_Hex8)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("HEX8", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tCoords = tMesh->Coordinates();
    auto tNVerts = tMesh->NumNodes();

    auto tDim = tMesh->NumDimensions();
    Kokkos::View<double*, MemSpace> tInField("not symmetric", tNVerts);
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    Kokkos::parallel_for("compute field", Kokkos::RangePolicy<OrdinalType>(0, tNVerts), KOKKOS_LAMBDA(OrdinalType iVertOrdinal)
    {
        tInField(iVertOrdinal) = tCoords(iVertOrdinal*tDim+2);
    });

    Kokkos::View<double*, MemSpace> tOutField("symmetric", tNVerts);
    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    for(OrdinalType i=0; i<tNVerts; i++)
    {
        if(tInField_host(i) > 1e-15)
        {
            if(tInField_host(i) < rz )
            {
                TEST_FLOATING_EQUALITY(2*rz-tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) > rz )
            {
                TEST_FLOATING_EQUALITY(tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) == 0.0 )
            {
                TEST_ASSERT(tOutField_host(i) == 0.0);
            }
        }
    }
  }

/******************************************************************************/
/*!
  \brief Enforce symmetry on a linear field on an asymmetric tet mesh.

  The linear tet mesh, while asymmetric, can approximate the symmetrized linear
  field accurately.  The test constructs a MeshMap with a SymmetryPlane:

    f(p(y)) = p(y)   if y >= 0.5
              p(1-y) if y < 0.5

  then constructs a field p(y) = y in {0.0,1.0} and applies the MeshMap, f(p)

  test passes if:
    f(p(y)) == y   for y > 0.5
    f(p(y)) == 1-y for y < 0.5
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryMeshMap_Quad4)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.5;
    double nx = 0.0, ny = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("QUAD4", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tCoords = tMesh->Coordinates();
    auto tNVerts = tMesh->NumNodes();

    auto tDim = tMesh->NumDimensions();
    Kokkos::View<double*, MemSpace> tInField("not symmetric", tNVerts);
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    Kokkos::parallel_for("compute field", Kokkos::RangePolicy<OrdinalType>(0, tNVerts), KOKKOS_LAMBDA(OrdinalType iVertOrdinal)
    {
        tInField(iVertOrdinal) = tCoords(iVertOrdinal*tDim+1);
    });

    Kokkos::View<double*, MemSpace> tOutField("symmetric", tNVerts);
    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    for(OrdinalType i=0; i<tNVerts; i++)
    {
        if(tInField_host(i) > 1e-15)
        {
            if(tInField_host(i) < ry )
            {
                TEST_FLOATING_EQUALITY(2*ry-tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) > ry )
            {
                TEST_FLOATING_EQUALITY(tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) == 0.0 )
            {
                TEST_ASSERT(tOutField_host(i) == 0.0);
            }
        }
    }
  }
/******************************************************************************/
/*!
  \brief Enforce symmetry on a uniform field on an asymmetric tet mesh.

  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  then constructs a field p(z) = 1 in {-0.5,0.5} and applies the MeshMap, f(p),
  as well as a linear filter, F.

  test passes if:
    F(f(p(z))) == 1 for all z
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryMeshMapWFilter)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <Filter>";
    input << "    <Type>Linear</Type>";
    input << "    <Radius>0.25</Radius>";
    input << "  </Filter>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tCoords = tMesh->Coordinates();
    auto tNVerts = tMesh->NumNodes();

    auto tDim = tMesh->NumDimensions();
    Kokkos::View<double*, MemSpace> tInField("uniform", tNVerts);
    Kokkos::View<double*, MemSpace> tOutField("also uniform", tNVerts);
    Kokkos::deep_copy(tInField, 1.0);

    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    for(OrdinalType i=0; i<tNVerts; i++)
    {
        TEST_FLOATING_EQUALITY(1.0, tOutField_host(i), tol_double);
    }
  }

/******************************************************************************/
/*!
  \brief Enforce symmetry on a uniform field on a hex8 mesh.

  The test constructs a MeshMap with a SymmetryPlane:

    f(p(x)) = p(x)   if x >= 0.5
              p(1-x) if x < 0.5

  then constructs a field
       p(x) = 1      if x >= 0.5
              0      if x < 0.5

  applies the MeshMap, f(p), as well as a linear filter, F.

  test passes if:
    F(f(p(x))) == 1 for all x
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryMeshMapWFilter_Hex8)
  {

    // create input for MeshMap
    //
    double rx = 0.5, ry = 0.0, rz = 0.0;
    double nx = 1.0, ny = 0.0, nz = 0.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <Filter>";
    input << "    <Type>Linear</Type>";
    input << "    <Radius>0.25</Radius>";
    input << "  </Filter>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("HEX8", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tNumDim = tMesh->NumDimensions();
    auto tCoords = tMesh->Coordinates();
    auto tConnect = tMesh->Connectivity();
    auto tNumNPE = tMesh->NumNodesPerElement();
    auto tNumNodes = tMesh->NumNodes();
    auto tNumElements = tMesh->NumElements();

    auto tDim = tMesh->NumDimensions();
    Kokkos::View<double*, MemSpace> tInField("uniform", tNumNodes);
    Kokkos::parallel_for("set field", Kokkos::RangePolicy<int>(0, tNumElements),
    KOKKOS_LAMBDA(int elemOrdinal)
    {
      bool tIntersected=false;
      for(Plato::OrdinalType iNode=0; iNode<tNumNPE; iNode++)
      {
        if(tCoords(tNumDim*tConnect(elemOrdinal*tNumNPE+iNode)) >= 0.5) tIntersected = true;
      }
      if(tIntersected)
      for(Plato::OrdinalType iNode=0; iNode<tNumNPE; iNode++)
      {
        tInField(tConnect(elemOrdinal*tNumNPE+iNode)) = 1.0;
      }
    });

    Kokkos::View<double*, MemSpace> tOutField("also uniform", tNumNodes);

    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    for(OrdinalType i=0; i<tNumNodes; i++)
    {
        TEST_FLOATING_EQUALITY(1.0, tOutField_host(i), tol_double);
    }
  }

/******************************************************************************/
/*!
  \brief Enforce symmetry on a uniform field on a quad4 mesh.

  The test constructs a MeshMap with a SymmetryPlane:

    f(p(x)) = p(x)   if x >= 0.5
              p(1-x) if x < 0.5

  then constructs a field
       p(x) = 1      if x >= 0.5
              0      if x < 0.5

  applies the MeshMap, f(p), as well as a linear filter, F.

  test passes if:
    F(f(p(x))) == 1 for all x
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryMeshMapWFilter_Quad4)
  {

    // create input for MeshMap
    //
    double rx = 0.5, ry = 0.0;
    double nx = 1.0, ny = 0.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <Filter>";
    input << "    <Type>Linear</Type>";
    input << "    <Radius>0.25</Radius>";
    input << "  </Filter>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("QUAD4", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tNumDim = tMesh->NumDimensions();
    auto tCoords = tMesh->Coordinates();
    auto tConnect = tMesh->Connectivity();
    auto tNumNPE = tMesh->NumNodesPerElement();
    auto tNumNodes = tMesh->NumNodes();
    auto tNumElements = tMesh->NumElements();

    auto tDim = tMesh->NumDimensions();
    Kokkos::View<double*, MemSpace> tInField("uniform", tNumNodes);
    Kokkos::parallel_for("set field", Kokkos::RangePolicy<int>(0, tNumElements),
    KOKKOS_LAMBDA(int elemOrdinal)
    {
      bool tIntersected=false;
      for(Plato::OrdinalType iNode=0; iNode<tNumNPE; iNode++)
      {
        if(tCoords(tNumDim*tConnect(elemOrdinal*tNumNPE+iNode)) >= 0.5) tIntersected = true;
      }
      if(tIntersected)
      for(Plato::OrdinalType iNode=0; iNode<tNumNPE; iNode++)
      {
        tInField(tConnect(elemOrdinal*tNumNPE+iNode)) = 1.0;
      }
    });

    Kokkos::View<double*, MemSpace> tOutField("also uniform", tNumNodes);

    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    for(OrdinalType i=0; i<tNumNodes; i++)
    {
        TEST_FLOATING_EQUALITY(1.0, tOutField_host(i), tol_double);
    }
  }
/******************************************************************************/
/*!
  \brief Enforce symmetry on a uniform field on an asymmetric tet mesh.

  The test constructs a MeshMap with a SymmetryPlane:

    f(p(x)) = p(x)   if x >= 0.5
              p(1-x) if x < 0.5

  then constructs a field
       p(x) = 1      if x >= 0.5
              0      if x < 0.5

  applies the MeshMap, f(p), as well as a linear filter, F.

  test passes if:
    F(f(p(x))) == 1 for all x
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryMeshMapWFilter_Tet4)
  {

    // create input for MeshMap
    //
    double rx = 0.5, ry = 0.0, rz = 0.0;
    double nx = 1.0, ny = 0.0, nz = 0.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <Filter>";
    input << "    <Type>Linear</Type>";
    input << "    <Radius>0.25</Radius>";
    input << "  </Filter>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tNumDim = tMesh->NumDimensions();
    auto tCoords = tMesh->Coordinates();
    auto tConnect = tMesh->Connectivity();
    auto tNumNPE = tMesh->NumNodesPerElement();
    auto tNumNodes = tMesh->NumNodes();
    auto tNumElements = tMesh->NumElements();

    auto tDim = tMesh->NumDimensions();
    Kokkos::View<double*, MemSpace> tInField("uniform", tNumNodes);
    Kokkos::parallel_for("set field", Kokkos::RangePolicy<int>(0, tNumElements),
    KOKKOS_LAMBDA(int elemOrdinal)
    {
      bool tIntersected=false;
      for(Plato::OrdinalType iNode=0; iNode<tNumNPE; iNode++)
      {
        if(tCoords(tNumDim*tConnect(elemOrdinal*tNumNPE+iNode)) >= 0.5) tIntersected = true;
      }
      if(tIntersected)
      for(Plato::OrdinalType iNode=0; iNode<tNumNPE; iNode++)
      {
        tInField(tConnect(elemOrdinal*tNumNPE+iNode)) = 1.0;
      }
    });

    Kokkos::View<double*, MemSpace> tOutField("also uniform", tNumNodes);

    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    for(OrdinalType i=0; i<tNumNodes; i++)
    {
        TEST_FLOATING_EQUALITY(1.0, tOutField_host(i), tol_double);
    }
  }
/******************************************************************************/
/*!
  \brief Test createTranspose() function in Plato::MeshMap.

  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  The map, f, and filter, F, are computed during construction.

  test passes if:
    (f^T)_{ij} = f_{ji}      : Transpose works
    (F^T)_{ij} = F_{ji}      : Transpose works
     F_{ii} = I              : Filter matrix rows sum to one
     F_{ij}!=0 if F_{ji}!=0  : Filter graph is symmetric
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, TransposeMatrix)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <Filter>";
    input << "    <Type>Linear</Type>";
    input << "    <Radius>0.25</Radius>";
    input << "  </Filter>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tMatrix  = to_full(tMeshMap->mMatrix);
    auto tMatrixT = to_full(tMeshMap->mMatrixT);

    double tol_double = 1e-12;
    for(int i=0; i<tMatrix.size(); i++)
    {
        for(int j=0; j<tMatrix[i].size(); j++)
        {
            TEST_FLOATING_EQUALITY(tMatrix[i][j], tMatrixT[j][i], tol_double);
        }
    }

    auto tFilter  = to_full(*(tMeshMap->mFilter));
    auto tFilterT = to_full(*(tMeshMap->mFilterT));

    std::vector<Plato::Scalar> tRowSum(tFilter.size());
    for(int i=0; i<tFilter.size(); i++)
    {
        tRowSum[i] = 0.0;
        for(int j=0; j<tFilter[i].size(); j++)
        {
            tRowSum[i] += tFilter[i][j];
            TEST_FLOATING_EQUALITY(tFilter[i][j], tFilterT[j][i], tol_double);
            if( tFilter[i][j] != 0.0 )
                TEST_ASSERT(tFilter[j][i] != 0.0);
        }
        TEST_FLOATING_EQUALITY(tRowSum[i], 1.0, tol_double);
    }

    auto tMatrixTT = tMeshMap->createTranspose(tMeshMap->mMatrixT);
    auto tMatrixTTF = to_full(tMatrixTT);

    for(int i=0; i<tMatrix.size(); i++)
    {
        for(int j=0; j<tMatrix[i].size(); j++)
        {
            TEST_FLOATING_EQUALITY(tMatrix[i][j], tMatrixTTF[i][j], tol_double);
        }
    }
  }

/******************************************************************************/
/*!
  \brief Test createTranspose() function in Plato::MeshMap.

  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  The map, f, and filter, F, are computed during construction.

  test passes if:
    (f^T)_{ij} = f_{ji}      : Transpose works
    (F^T)_{ij} = F_{ji}      : Transpose works
     F_{ii} = I              : Filter matrix rows sum to one
     F_{ij}!=0 if F_{ji}!=0  : Filter graph is symmetric
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, TransposeMatrix_Tet10)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <Filter>";
    input << "    <Type>Linear</Type>";
    input << "    <Radius>0.25</Radius>";
    input << "  </Filter>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(tMesh, tMeshMapParams);

    auto tMatrix  = to_full(tMeshMap->mMatrix);
    auto tMatrixT = to_full(tMeshMap->mMatrixT);

    double tol_double = 1e-12;
    for(int i=0; i<tMatrix.size(); i++)
    {
        for(int j=0; j<tMatrix[i].size(); j++)
        {
            TEST_FLOATING_EQUALITY(tMatrix[i][j], tMatrixT[j][i], tol_double);
        }
    }

    auto tFilter  = to_full(*(tMeshMap->mFilter));
    auto tFilterT = to_full(*(tMeshMap->mFilterT));

    std::vector<Plato::Scalar> tRowSum(tFilter.size());
    for(int i=0; i<tFilter.size(); i++)
    {
        tRowSum[i] = 0.0;
        for(int j=0; j<tFilter[i].size(); j++)
        {
            tRowSum[i] += tFilter[i][j];
            TEST_FLOATING_EQUALITY(tFilter[i][j], tFilterT[j][i], tol_double);
            if( tFilter[i][j] != 0.0 )
                TEST_ASSERT(tFilter[j][i] != 0.0);
        }
        TEST_FLOATING_EQUALITY(tRowSum[i], 1.0, tol_double);
    }

    auto tMatrixTT = tMeshMap->createTranspose(tMeshMap->mMatrixT);
    auto tMatrixTTF = to_full(tMatrixTT);

    for(int i=0; i<tMatrix.size(); i++)
    {
        for(int j=0; j<tMatrix[i].size(); j++)
        {
            TEST_FLOATING_EQUALITY(tMatrix[i][j], tMatrixTTF[i][j], tol_double);
        }
    }
  }

} // namespace PlatoTestMeshMap
