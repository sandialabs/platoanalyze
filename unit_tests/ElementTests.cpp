#include "util/PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "PlatoStaticsTypes.hpp"
#include "MechanicsElement.hpp"
#include "PlatoMeshExpr.hpp"
#include "WorksetBase.hpp"
#include "Hex8.hpp"
#include "Hex27.hpp"
#include "Tet10.hpp"
#include "Tet4.hpp"
#include "Tri6.hpp"
#include "Tri3.hpp"

#include "SurfaceArea.hpp"

using ordType = typename Plato::ScalarMultiVector::size_type;


/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Tet4> cubature weights
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet4, MechanicsTet4_CubWeightsSum )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tet4>;

    auto tCubWeights = ElementType::getCubWeights();

    Plato::Scalar tTotalWeight(0.0);
    for(int iWeight=0; iWeight<tCubWeights.size(); iWeight++)
    {
        tTotalWeight += tCubWeights(iWeight);
    }

    TEST_FLOATING_EQUALITY(tTotalWeight, Plato::Scalar(1)/6, 1e-13);
}


/******************************************************************************/
/*! 
  \brief Check the Hex27 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex27, Hex27_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Hex27::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Hex27::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Hex27::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 27);
    TEST_ASSERT(tNodesPerFace  == 9 );
    TEST_ASSERT(tSpaceDims     == 3 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Hex27> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex27, MechanicsHex27_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Hex27>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 27);
    TEST_ASSERT(tNodesPerFace  == 9 );
    TEST_ASSERT(tDofsPerCell   == 81);
    TEST_ASSERT(tDofsPerNode   == 3 );
    TEST_ASSERT(tSpaceDims     == 3 );
    TEST_ASSERT(tNumVoigtTerms == 6 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Hex27> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex27, MechanicsHex27_CubWeightsSum )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Hex27>;

    auto tCubWeights = ElementType::getCubWeights();

    Plato::Scalar tTotalWeight(0.0);
    for(int iWeight=0; iWeight<tCubWeights.size(); iWeight++)
    {
        tTotalWeight += tCubWeights(iWeight);
    }

    TEST_FLOATING_EQUALITY(tTotalWeight, 8.0, 1e-13);
}

/******************************************************************************/
/*! 
  \brief Evaluate the Hex27 basis functions at each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex27, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", Plato::Hex27::mNumNodesPerCell, Plato::Hex27::mNumNodesPerCell);

    Plato::Matrix<Plato::Hex27::mNumNodesPerCell, Plato::Hex27::mNumSpatialDims> tPoints = {
        -1.0, -1.0, -1.0,
         1.0, -1.0, -1.0,
         1.0,  1.0, -1.0,
        -1.0,  1.0, -1.0,
        -1.0, -1.0,  1.0,
         1.0, -1.0,  1.0,
         1.0,  1.0,  1.0,
        -1.0,  1.0,  1.0,
         0.0, -1.0, -1.0,
         1.0,  0.0, -1.0,
         0.0,  1.0, -1.0,
        -1.0,  0.0, -1.0,
        -1.0, -1.0,  0.0,
         1.0, -1.0,  0.0,
         1.0,  1.0,  0.0,
        -1.0,  1.0,  0.0,
         0.0, -1.0,  1.0,
         1.0,  0.0,  1.0,
         0.0,  1.0,  1.0,
        -1.0,  0.0,  1.0,
         0.0,  0.0,  0.0,
         0.0,  0.0, -1.0,
         0.0,  0.0,  1.0,
        -1.0,  0.0,  0.0,
         1.0,  0.0,  0.0,
         0.0, -1.0,  0.0,
         0.0,  1.0,  0.0
    };


    Kokkos::parallel_for("basis functions", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        for(ordType I=0; I<Plato::Hex27::mNumNodesPerCell; I++)
        {
            auto tValues = Plato::Hex27::basisValues(tPoints(I));
            for(ordType i=0; i<Plato::Hex27::mNumNodesPerCell; i++)
            {
                tValuesView(I,i) = tValues(i);
            }
        }
    });

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    for(int i=0; i<Plato::Hex27::mNumNodesPerCell; i++)
    {
        for(int j=0; j<Plato::Hex27::mNumNodesPerCell; j++)
        {
            if ( i==j )
            {
                TEST_ASSERT(tValuesHost(i,j) == 1);
            } else {
                TEST_ASSERT(tValuesHost(i,j) == 0);
            }
        }
    }
}
/******************************************************************************/
/*! 
  \brief Evaluate the Hex27 basis function gradients at (0.25, 0.25, 0.25)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex27, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Hex27::mNumNodesPerCell, Plato::Hex27::mNumSpatialDims);

    Kokkos::parallel_for("basis function derivatives", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Hex27::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25; tPoint(2) = 0.25;
        auto tGrads = Plato::Hex27::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Hex27::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Hex27::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    });

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = {
      {-Plato::Scalar(9)/4096,   -Plato::Scalar(9)/4096,   -Plato::Scalar(9)/4096},
      { Plato::Scalar(27)/4096,   Plato::Scalar(15)/4096,   Plato::Scalar(15)/4096},
      {-Plato::Scalar(45)/4096,  -Plato::Scalar(45)/4096,  -Plato::Scalar(25)/4096},
      { Plato::Scalar(15)/4096,   Plato::Scalar(27)/4096,   Plato::Scalar(15)/4096},
      { Plato::Scalar(15)/4096,   Plato::Scalar(15)/4096,   Plato::Scalar(27)/4096}, 
      {-Plato::Scalar(45)/4096,  -Plato::Scalar(25)/4096,  -Plato::Scalar(45)/4096},
      { Plato::Scalar(75)/4096,   Plato::Scalar(75)/4096,   Plato::Scalar(75)/4096},
      {-Plato::Scalar(25)/4096,  -Plato::Scalar(45)/4096,  -Plato::Scalar(45)/4096},
      {-Plato::Scalar(9)/2048,    Plato::Scalar(45)/2048,   Plato::Scalar(45)/2048},
      {-Plato::Scalar(135)/2048,  Plato::Scalar(15)/2048,  -Plato::Scalar(75)/2048},
      { Plato::Scalar(15)/2048,  -Plato::Scalar(135)/2048, -Plato::Scalar(75)/2048},
      { Plato::Scalar(45)/2048,  -Plato::Scalar(9)/2048,    Plato::Scalar(45)/2048},
      { Plato::Scalar(45)/2048,   Plato::Scalar(45)/2048,  -Plato::Scalar(9)/2048},
      {-Plato::Scalar(135)/2048, -Plato::Scalar(75)/2048,   Plato::Scalar(15)/2048},
      { Plato::Scalar(225)/2048,  Plato::Scalar(225)/2048, -Plato::Scalar(25)/2048},
      {-Plato::Scalar(75)/2048,  -Plato::Scalar(135)/2048,  Plato::Scalar(15)/2048},
      { Plato::Scalar(15)/2048,  -Plato::Scalar(75)/2048,  -Plato::Scalar(135)/2048},
      { Plato::Scalar(225)/2048, -Plato::Scalar(25)/2048,   Plato::Scalar(225)/2048},
      {-Plato::Scalar(25)/2048,   Plato::Scalar(225)/2048,  Plato::Scalar(225)/2048},
      {-Plato::Scalar(75)/2048,   Plato::Scalar(15)/2048,  -Plato::Scalar(135)/2048},
      {-Plato::Scalar(225)/512,  -Plato::Scalar(225)/512,  -Plato::Scalar(225)/512},
      { Plato::Scalar(45)/1024,   Plato::Scalar(45)/1024,  -Plato::Scalar(225)/1024},
      {-Plato::Scalar(75)/1024,  -Plato::Scalar(75)/1024,   Plato::Scalar(675)/1024},
      {-Plato::Scalar(225)/1024,  Plato::Scalar(45)/1024,   Plato::Scalar(45)/1024},
      { Plato::Scalar(675)/1024, -Plato::Scalar(75)/1024,  -Plato::Scalar(75)/1024},
      { Plato::Scalar(45)/1024,  -Plato::Scalar(225)/1024,  Plato::Scalar(45)/1024},
      {-Plato::Scalar(75)/1024,   Plato::Scalar(675)/1024, -Plato::Scalar(75)/1024}
    };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Check the Hex8 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex8, Hex8_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Hex8::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Hex8::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Hex8::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 8 );
    TEST_ASSERT(tNodesPerFace  == 4 );
    TEST_ASSERT(tSpaceDims     == 3 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Hex8> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex8, MechanicsHex8_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Hex8>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 8 );
    TEST_ASSERT(tNodesPerFace  == 4 );
    TEST_ASSERT(tDofsPerCell   == 24);
    TEST_ASSERT(tDofsPerNode   == 3 );
    TEST_ASSERT(tSpaceDims     == 3 );
    TEST_ASSERT(tNumVoigtTerms == 6 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Hex8> cubature weights
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex8, MechanicsHex8_CubWeightsSum )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Hex8>;

    auto tCubWeights = ElementType::getCubWeights();

    Plato::Scalar tTotalWeight(0.0);
    for(int iWeight=0; iWeight<tCubWeights.size(); iWeight++)
    {
        tTotalWeight += tCubWeights(iWeight);
    }

    TEST_FLOATING_EQUALITY(tTotalWeight, 8.0, 1e-13);
}

/******************************************************************************/
/*! 
  \brief Evaluate the Hex8 basis functions at each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex8, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", 8, Plato::Hex8::mNumNodesPerCell);

    Kokkos::parallel_for("basis functions", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Hex8::mNumSpatialDims> tPoint;

        tPoint(0) = -1.0; tPoint(1) = -1.0; tPoint(2) = -1.0;
        auto tValues = Plato::Hex8::basisValues(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++) { tValuesView(0,i) = tValues(i); }

        tPoint(0) = 1.0; tPoint(1) = -1.0; tPoint(2) = -1.0;
        tValues = Plato::Hex8::basisValues(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++) { tValuesView(1,i) = tValues(i); }

        tPoint(0) = 1.0; tPoint(1) = 1.0; tPoint(2) = -1.0;
        tValues = Plato::Hex8::basisValues(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++) { tValuesView(2,i) = tValues(i); }

        tPoint(0) = -1.0; tPoint(1) = 1.0; tPoint(2) = -1.0;
        tValues = Plato::Hex8::basisValues(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++) { tValuesView(3,i) = tValues(i); }

        tPoint(0) = -1.0; tPoint(1) = -1.0; tPoint(2) = 1.0;
        tValues = Plato::Hex8::basisValues(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++) { tValuesView(4,i) = tValues(i); }

        tPoint(0) = 1.0; tPoint(1) = -1.0; tPoint(2) = 1.0;
        tValues = Plato::Hex8::basisValues(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++) { tValuesView(5,i) = tValues(i); }

        tPoint(0) = 1.0; tPoint(1) = 1.0; tPoint(2) = 1.0;
        tValues = Plato::Hex8::basisValues(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++) { tValuesView(6,i) = tValues(i); }

        tPoint(0) = -1.0; tPoint(1) = 1.0; tPoint(2) = 1.0;
        tValues = Plato::Hex8::basisValues(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++) { tValuesView(7,i) = tValues(i); }

    });

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    std::vector<std::vector<Plato::Scalar>> tValuesGold = {
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
    };

    int tNumGold_I=tValuesGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tValuesGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tValuesHost(i,j), tValuesGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Hex8 basis function gradients at (0.25, 0.25, 0.25)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex8, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Hex8::mNumNodesPerCell, Plato::Hex8::mNumSpatialDims);

    Kokkos::parallel_for("basis function derivatives", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Hex8::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25; tPoint(2) = 0.25;
        auto tGrads = Plato::Hex8::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Hex8::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Hex8::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    });

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = {
      {-Plato::Scalar(9)/128,  -Plato::Scalar(9)/128,  -Plato::Scalar(9)/128 },
      { Plato::Scalar(9)/128,  -Plato::Scalar(15)/128, -Plato::Scalar(15)/128},
      { Plato::Scalar(15)/128,  Plato::Scalar(15)/128, -Plato::Scalar(25)/128},
      {-Plato::Scalar(15)/128,  Plato::Scalar(9)/128,  -Plato::Scalar(15)/128},
      {-Plato::Scalar(15)/128, -Plato::Scalar(15)/128,  Plato::Scalar(9)/128 },
      { Plato::Scalar(15)/128, -Plato::Scalar(25)/128,  Plato::Scalar(15)/128},
      { Plato::Scalar(25)/128,  Plato::Scalar(25)/128,  Plato::Scalar(25)/128},
      {-Plato::Scalar(25)/128,  Plato::Scalar(15)/128,  Plato::Scalar(15)/128}
    };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Hex8 jacobian at (0.25, 0.25, 0.25) for a cell that's
  in the reference configuration.  Jacobian should be identity.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Hex8, JacobianParentCoords )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Hex8>;

    Plato::ScalarMultiVector tJacobianView("jacobian", Plato::Hex8::mNumSpatialDims, Plato::Hex8::mNumSpatialDims);
    Plato::ScalarArray3D tConfig("node locations", 1, Plato::Hex8::mNumNodesPerCell, Plato::Hex8::mNumSpatialDims);

    Kokkos::parallel_for("cell jacobian", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        tConfig(0,0,0) = -1.0; tConfig(0,0,1) = -1.0; tConfig(0,0,2) = -1.0;
        tConfig(0,1,0) =  1.0; tConfig(0,1,1) = -1.0; tConfig(0,1,2) = -1.0;
        tConfig(0,2,0) =  1.0; tConfig(0,2,1) =  1.0; tConfig(0,2,2) = -1.0;
        tConfig(0,3,0) = -1.0; tConfig(0,3,1) =  1.0; tConfig(0,3,2) = -1.0;
        tConfig(0,4,0) = -1.0; tConfig(0,4,1) = -1.0; tConfig(0,4,2) =  1.0;
        tConfig(0,5,0) =  1.0; tConfig(0,5,1) = -1.0; tConfig(0,5,2) =  1.0;
        tConfig(0,6,0) =  1.0; tConfig(0,6,1) =  1.0; tConfig(0,6,2) =  1.0;
        tConfig(0,7,0) = -1.0; tConfig(0,7,1) =  1.0; tConfig(0,7,2) =  1.0;

        Plato::Array<ElementType::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25; tPoint(2) = 0.25;
        auto tJacobian = ElementType::jacobian(tPoint, tConfig, ordinal);
        for(ordType i=0; i<Plato::Hex8::mNumSpatialDims; i++)
        {
            for(ordType j=0; j<Plato::Hex8::mNumSpatialDims; j++)
            {
                tJacobianView(i,j) = tJacobian(i,j);
            }
        }
    });

    auto tJacobianHost = Kokkos::create_mirror_view( tJacobianView );
    Kokkos::deep_copy( tJacobianHost, tJacobianView );

    std::vector<std::vector<Plato::Scalar>> tJacobianGold = { {1, 0, 0}, { 0, 1, 0 }, {0, 0, 1} };

    int tNumGold_I=tJacobianGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tJacobianGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tJacobianHost(i,j), tJacobianGold[i][j], 1e-13);
        }
    }
}


/******************************************************************************/
/*! 
  \brief Check the Quad9 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Quad9, Quad9_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Quad9::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Quad9::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Quad9::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 9 );
    TEST_ASSERT(tNodesPerFace  == 3 );
    TEST_ASSERT(tSpaceDims     == 2 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Quad9> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Quad9, MechanicsQuad9_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Quad9>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 9 );
    TEST_ASSERT(tNodesPerFace  == 3 );
    TEST_ASSERT(tDofsPerCell   == 18);
    TEST_ASSERT(tDofsPerNode   == 2 );
    TEST_ASSERT(tSpaceDims     == 2 );
    TEST_ASSERT(tNumVoigtTerms == 3 );
}
/******************************************************************************/
/*! 
  \brief Evaluate the Quad9 basis functions at each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Quad9, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", Plato::Quad9::mNumNodesPerCell, Plato::Quad9::mNumNodesPerCell);

    Plato::Matrix<Plato::Quad9::mNumNodesPerCell, Plato::Quad9::mNumSpatialDims> tPoints = {
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0,
         0.0, -1.0,
         1.0,  0.0,
         0.0,  1.0,
        -1.0,  0.0,
         0.0,  0.0
    };


    Kokkos::parallel_for("basis functions", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        for(ordType I=0; I<Plato::Quad9::mNumNodesPerCell; I++)
        {
            auto tValues = Plato::Quad9::basisValues(tPoints(I));
            for(ordType i=0; i<Plato::Quad9::mNumNodesPerCell; i++)
            {
                tValuesView(I,i) = tValues(i);
            }
        }
    });

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    for(int i=0; i<Plato::Quad9::mNumNodesPerCell; i++)
    {
        for(int j=0; j<Plato::Quad9::mNumNodesPerCell; j++)
        {
            if ( i==j )
            {
                TEST_ASSERT(tValuesHost(i,j) == 1);
            } else {
                TEST_ASSERT(tValuesHost(i,j) == 0);
            }
        }
    }
}
/******************************************************************************/
/*! 
  \brief Evaluate the Quad9 basis function gradients at (0.25, 0.25)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Quad9, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Quad9::mNumNodesPerCell, Plato::Quad9::mNumSpatialDims);

    Kokkos::parallel_for("basis function derivatives", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Quad9::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25;
        auto tGrads = Plato::Quad9::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Quad9::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Quad9::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    });

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = {
      { Plato::Scalar(3)/128,  Plato::Scalar(3)/128  },
      {-Plato::Scalar(9)/128, -Plato::Scalar(5)/128  },
      { Plato::Scalar(15)/128, Plato::Scalar(15)/128 },
      {-Plato::Scalar(5)/128, -Plato::Scalar(9)/128  },
      { Plato::Scalar(3)/64,  -Plato::Scalar(15)/64  },
      { Plato::Scalar(45)/64, -Plato::Scalar(5)/64   },
      {-Plato::Scalar(5)/64,   Plato::Scalar(45)/64  },
      {-Plato::Scalar(15)/64,  Plato::Scalar(3)/64   },
      {-Plato::Scalar(15)/32, -Plato::Scalar(15)/32  }
    };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Check the Quad4 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Quad4, Quad4_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Quad4::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Quad4::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Quad4::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 4 );
    TEST_ASSERT(tNodesPerFace  == 2 );
    TEST_ASSERT(tSpaceDims     == 2 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Quad4> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Quad4, MechanicsQuad4_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Quad4>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 4 );
    TEST_ASSERT(tNodesPerFace  == 2 );
    TEST_ASSERT(tDofsPerCell   == 8 );
    TEST_ASSERT(tDofsPerNode   == 2 );
    TEST_ASSERT(tSpaceDims     == 2 );
    TEST_ASSERT(tNumVoigtTerms == 3 );
}

/******************************************************************************/
/*! 
  \brief Evaluate the Quad4 basis functions at each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Quad4, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", Plato::Quad4::mNumNodesPerCell, Plato::Quad4::mNumNodesPerCell);

    Plato::Matrix<Plato::Quad4::mNumNodesPerCell, Plato::Quad4::mNumSpatialDims> tPoints = {
        -1.0, -1.0,
         1.0, -1.0,
         1.0,  1.0,
        -1.0,  1.0
    };


    Kokkos::parallel_for("basis functions", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        for(ordType I=0; I<Plato::Quad4::mNumNodesPerCell; I++)
        {
            auto tValues = Plato::Quad4::basisValues(tPoints(I));
            for(ordType i=0; i<Plato::Quad4::mNumNodesPerCell; i++)
            {
                tValuesView(I,i) = tValues(i);
            }
        }
    });

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    for(int i=0; i<Plato::Quad4::mNumNodesPerCell; i++)
    {
        for(int j=0; j<Plato::Quad4::mNumNodesPerCell; j++)
        {
            if ( i==j )
            {
                TEST_ASSERT(tValuesHost(i,j) == 1);
            } else {
                TEST_ASSERT(tValuesHost(i,j) == 0);
            }
        }
    }
}
/******************************************************************************/
/*! 
  \brief Evaluate the Quad4 basis function gradients at (0.25, 0.25)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Quad4, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Quad4::mNumNodesPerCell, Plato::Quad4::mNumSpatialDims);

    Kokkos::parallel_for("basis function derivatives", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Quad4::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25;
        auto tGrads = Plato::Quad4::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Quad4::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Quad4::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    });

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = {
      {-Plato::Scalar(3)/16, -Plato::Scalar(3)/16 },
      { Plato::Scalar(3)/16, -Plato::Scalar(5)/16 },
      { Plato::Scalar(5)/16,  Plato::Scalar(5)/16 },
      {-Plato::Scalar(5)/16,  Plato::Scalar(3)/16 }
    };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Check the Bar2 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Bar2, Bar2_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Bar2::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Bar2::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Bar2::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 2 );
    TEST_ASSERT(tNodesPerFace  == 1 );
    TEST_ASSERT(tSpaceDims     == 1 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Bar2> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Bar2, MechanicsBar2_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Bar2>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 2 );
    TEST_ASSERT(tNodesPerFace  == 1 );
    TEST_ASSERT(tDofsPerCell   == 2 );
    TEST_ASSERT(tDofsPerNode   == 1 );
    TEST_ASSERT(tSpaceDims     == 1 );
    TEST_ASSERT(tNumVoigtTerms == 1 );
}

/******************************************************************************/
/*! 
  \brief Evaluate the Bar2 basis functions at each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Bar2, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", Plato::Bar2::mNumNodesPerCell, Plato::Bar2::mNumNodesPerCell);

    Plato::Matrix<Plato::Bar2::mNumNodesPerCell, Plato::Bar2::mNumSpatialDims> tPoints = { -1.0, 1.0 };


    Kokkos::parallel_for("basis functions", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        for(ordType I=0; I<Plato::Bar2::mNumNodesPerCell; I++)
        {
            auto tValues = Plato::Bar2::basisValues(tPoints(I));
            for(ordType i=0; i<Plato::Bar2::mNumNodesPerCell; i++)
            {
                tValuesView(I,i) = tValues(i);
            }
        }
    });

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    for(int i=0; i<Plato::Bar2::mNumNodesPerCell; i++)
    {
        for(int j=0; j<Plato::Bar2::mNumNodesPerCell; j++)
        {
            if ( i==j )
            {
                TEST_ASSERT(tValuesHost(i,j) == 1);
            } else {
                TEST_ASSERT(tValuesHost(i,j) == 0);
            }
        }
    }
}
/******************************************************************************/
/*! 
  \brief Evaluate the Bar2 basis function gradients at (0.25, 0.25)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Bar2, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Bar2::mNumNodesPerCell, Plato::Bar2::mNumSpatialDims);

    Kokkos::parallel_for("basis function derivatives", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Bar2::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25;
        auto tGrads = Plato::Bar2::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Bar2::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Bar2::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    });

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = { {Plato::Scalar(-1)/2}, { Plato::Scalar(1)/2} };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Check the Tet10 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, Tet10_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Tet10::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Tet10::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Tet10::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 10);
    TEST_ASSERT(tNodesPerFace  == 6 );
    TEST_ASSERT(tSpaceDims     == 3 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Tet10> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, MechanicsTet10_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 10);
    TEST_ASSERT(tNodesPerFace  == 6 );
    TEST_ASSERT(tDofsPerCell   == 30);
    TEST_ASSERT(tDofsPerNode   == 3 );
    TEST_ASSERT(tSpaceDims     == 3 );
    TEST_ASSERT(tNumVoigtTerms == 6 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Tet10> cubature weights
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, MechanicsTet10_CubWeightsSum )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    auto tCubWeights = ElementType::getCubWeights();

    Plato::Scalar tTotalWeight(0.0);
    for(int iWeight=0; iWeight<tCubWeights.size(); iWeight++)
    {
        tTotalWeight += tCubWeights(iWeight);
    }

    TEST_FLOATING_EQUALITY(tTotalWeight, Plato::Scalar(1)/6, 1e-13);
}

/******************************************************************************/
/*! 
  \brief Check the Tri3 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri3, Tri3_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Tri3::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Tri3::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Tri3::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 3 );
    TEST_ASSERT(tNodesPerFace  == 2 );
    TEST_ASSERT(tSpaceDims     == 2 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Tri3> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri3, MechanicsTri3_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tri3>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 3 );
    TEST_ASSERT(tNodesPerFace  == 2 );
    TEST_ASSERT(tDofsPerCell   == 6 );
    TEST_ASSERT(tDofsPerNode   == 2 );
    TEST_ASSERT(tSpaceDims     == 2 );
    TEST_ASSERT(tNumVoigtTerms == 3 );
}
/******************************************************************************/
/*! 
  \brief Evaluate the Tri3 basis functions at each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri3, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", Plato::Tri3::mNumNodesPerCell, Plato::Tri3::mNumNodesPerCell);

    Plato::Matrix<Plato::Tri3::mNumNodesPerCell, Plato::Tri3::mNumSpatialDims> tPoints = {
         0.0,  0.0,
         1.0,  0.0,
         0.0,  1.0
    };


    Kokkos::parallel_for("basis functions", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        for(ordType I=0; I<Plato::Tri3::mNumNodesPerCell; I++)
        {
            auto tValues = Plato::Tri3::basisValues(tPoints(I));
            for(ordType i=0; i<Plato::Tri3::mNumNodesPerCell; i++)
            {
                tValuesView(I,i) = tValues(i);
            }
        }
    });

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    for(int i=0; i<Plato::Tri3::mNumNodesPerCell; i++)
    {
        for(int j=0; j<Plato::Tri3::mNumNodesPerCell; j++)
        {
            if ( i==j )
            {
                TEST_ASSERT(tValuesHost(i,j) == 1);
            } else {
                TEST_ASSERT(tValuesHost(i,j) == 0);
            }
        }
    }
}
/******************************************************************************/
/*! 
  \brief Evaluate the Tri3 basis function gradients at (0.25, 0.25)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri3, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Tri3::mNumNodesPerCell, Plato::Tri3::mNumSpatialDims);

    Kokkos::parallel_for("basis function derivatives", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Tri3::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25;
        auto tGrads = Plato::Tri3::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Tri3::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Tri3::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    });

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = { {-1, -1}, {1, 0}, {0, 1} };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}


/******************************************************************************/
/*! 
  \brief Check the Tri6 constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, Tri6_Constants )
{ 
    constexpr auto tNodesPerCell  = Plato::Tri6::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = Plato::Tri6::mNumNodesPerFace;
    constexpr auto tSpaceDims     = Plato::Tri6::mNumSpatialDims;

    TEST_ASSERT(tNodesPerCell  == 6 );
    TEST_ASSERT(tNodesPerFace  == 3 );
    TEST_ASSERT(tSpaceDims     == 2 );
}

/******************************************************************************/
/*! 
  \brief Check the MechanicsElement<Tri6> constants
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, MechanicsTri6_Constants )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tri6>;

    constexpr auto tNodesPerCell  = ElementType::mNumNodesPerCell;
    constexpr auto tNodesPerFace  = ElementType::mNumNodesPerFace;
    constexpr auto tDofsPerCell   = ElementType::mNumDofsPerCell;
    constexpr auto tDofsPerNode   = ElementType::mNumDofsPerNode;
    constexpr auto tSpaceDims     = ElementType::mNumSpatialDims;
    constexpr auto tNumVoigtTerms = ElementType::mNumVoigtTerms;

    TEST_ASSERT(tNodesPerCell  == 6 );
    TEST_ASSERT(tNodesPerFace  == 3 );
    TEST_ASSERT(tDofsPerCell   == 12);
    TEST_ASSERT(tDofsPerNode   == 2 );
    TEST_ASSERT(tSpaceDims     == 2 );
    TEST_ASSERT(tNumVoigtTerms == 3 );
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tet10 basis functions at each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", Plato::Tet10::mNumNodesPerCell, Plato::Tet10::mNumNodesPerCell);

    Plato::Matrix<Plato::Tet10::mNumNodesPerCell, Plato::Tet10::mNumSpatialDims> tPoints = {
         0.0,  0.0,  0.0,
         1.0,  0.0,  0.0,
         0.0,  1.0,  0.0,
         0.0,  0.0,  1.0,
         0.5,  0.0,  0.0,
         0.5,  0.5,  0.0,
         0.0,  0.5,  0.0,
         0.0,  0.0,  0.5,
         0.5,  0.0,  0.5,
         0.0,  0.5,  0.5,
    };


    Kokkos::parallel_for("basis functions", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        for(ordType I=0; I<Plato::Tet10::mNumNodesPerCell; I++)
        {
            auto tValues = Plato::Tet10::basisValues(tPoints(I));
            for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++)
            {
                tValuesView(I,i) = tValues(i);
            }
        }
    });

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    for(int i=0; i<Plato::Tet10::mNumNodesPerCell; i++)
    {
        for(int j=0; j<Plato::Tet10::mNumNodesPerCell; j++)
        {
            if ( i==j )
            {
                TEST_ASSERT(tValuesHost(i,j) == 1);
            } else {
                TEST_ASSERT(tValuesHost(i,j) == 0);
            }
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tri6 basis functions at (1/6, 1/6) and at
         each node location.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, BasisFunctions )
{ 
    Plato::ScalarMultiVector tValuesView("basis values", 7, Plato::Tri6::mNumNodesPerCell);

    Kokkos::parallel_for("basis functions", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Tri6::mNumSpatialDims> tPoint;

        tPoint(0) = Plato::Scalar(1)/6; tPoint(1) = Plato::Scalar(1)/6;
        auto tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(0,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.0;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(1,i) = tValues(i); }

        tPoint(0) = 1.0; tPoint(1) = 0.0;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(2,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 1.0;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(3,i) = tValues(i); }

        tPoint(0) = 0.5; tPoint(1) = 0.0;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(4,i) = tValues(i); }

        tPoint(0) = 0.5; tPoint(1) = 0.5;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(5,i) = tValues(i); }

        tPoint(0) = 0.0; tPoint(1) = 0.5;
        tValues = Plato::Tri6::basisValues(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++) { tValuesView(6,i) = tValues(i); }

    });

    auto tValuesHost = Kokkos::create_mirror_view( tValuesView );
    Kokkos::deep_copy( tValuesHost, tValuesView );

    std::vector<std::vector<Plato::Scalar>> tValuesGold = {
      {2.0/9.0, -1.0/9.0, -1.0/9.0, 4.0/9.0, 1.0/9.0, 4.0/9.0},
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
    };

    int tNumGold_I=tValuesGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tValuesGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tValuesHost(i,j), tValuesGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tet10 basis function gradients at (0.25, 0.25, 0.25)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Tet10::mNumNodesPerCell, Plato::Tet10::mNumSpatialDims);

    Kokkos::parallel_for("basis function derivatives", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Tet10::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25; tPoint(2) = 0.25;
        auto tGrads = Plato::Tet10::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Tet10::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Tet10::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    });

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = {
      { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}, { 0, -1, -1},
      { 1,  1,  0}, {-1,  0, -1}, {-1, -1,  0}, { 1,  0,  1}, { 0,  1,  1}
    };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Integrate an expression over the mesh domain
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, IntegrateExpression )
{ 
  constexpr int meshWidth=1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TET10", meshWidth);

  using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

  Plato::WorksetBase<ElementType> worksetBase(tMesh);

  auto tNumCells     = tMesh->NumElements();
  auto tNodesPerCell = ElementType::mNumNodesPerCell;
  auto tSpaceDims    = ElementType::mNumSpatialDims;

  auto tCubPoints = ElementType::getCubPoints();
  auto tCubWeights = ElementType::getCubWeights();
  auto tNumPoints = tCubWeights.size();

  Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDims);

  worksetBase.worksetConfig(tConfigWS);

  // map points to physical space
  //
  Plato::ScalarArray3DT<Plato::Scalar> tPhysicalPoints("cub points physical space", tNumCells, tNumPoints, tSpaceDims);

  Plato::mapPoints<ElementType>(tConfigWS, tPhysicalPoints);

  // get integrand values at quadrature points
  //
  std::string tFuncString = "1.0";
  auto tTotalNumPoints = tNumCells*tNumPoints;
  Plato::ScalarMultiVectorT<Plato::Scalar> tFxnValues("function values", tTotalNumPoints, 1);
  Plato:: getFunctionValues<ElementType::mNumSpatialDims>(tPhysicalPoints, tFuncString, tFxnValues);

  Plato::Scalar tSum(0.0);
  Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, tNumPoints}),
  KOKKOS_LAMBDA(const int & cellOrdinal, const int & ptOrdinal, Plato::Scalar & aUpdate)
  {
      auto tCubPoint  = tCubPoints(ptOrdinal);
      auto tCubWeight = tCubWeights(ptOrdinal);

      auto tJacobian = ElementType::jacobian(tCubPoint, tConfigWS, cellOrdinal);

      auto tCellVolume = Plato::determinant(tJacobian);

      auto tPointValue = tFxnValues(cellOrdinal*tNumPoints+ptOrdinal, 0);
      aUpdate += tCellVolume*tPointValue*tCubWeight;
  }, tSum);

  TEST_FLOATING_EQUALITY(tSum, 1.0, 1e-13);

}

/******************************************************************************/
/*! 
  \brief Evaluate the Tri6 basis function gradients at (1/6, 1/6)
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, BasisFunctionGradients )
{ 
    Plato::ScalarMultiVector tGradsView("basis grads", Plato::Tri6::mNumNodesPerCell, Plato::Tri6::mNumSpatialDims);

    Kokkos::parallel_for("basis function derivatives", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        Plato::Array<Plato::Tri6::mNumSpatialDims> tPoint;

        tPoint(0) = Plato::Scalar(1)/6; tPoint(1) = Plato::Scalar(1)/6;
        auto tGrads = Plato::Tri6::basisGrads(tPoint);
        for(ordType i=0; i<Plato::Tri6::mNumNodesPerCell; i++)
        {
            for(ordType j=0; j<Plato::Tri6::mNumSpatialDims; j++)
            {
                tGradsView(i,j) = tGrads(i,j);
            }
        }
    });

    auto tGradsHost = Kokkos::create_mirror_view( tGradsView );
    Kokkos::deep_copy( tGradsHost, tGradsView );

    std::vector<std::vector<Plato::Scalar>> tGradsGold = {
        {Plato::Scalar(-5)/3, Plato::Scalar(-5)/3},
        {Plato::Scalar(-1)/3, 0                  },
        {0,                   Plato::Scalar(-1)/3},
        {2,                   Plato::Scalar(-2)/3},
        {Plato::Scalar(2)/3,  Plato::Scalar(2)/3 },
        {Plato::Scalar(-2)/3, 2                  }
    };

    int tNumGold_I=tGradsGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tGradsGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tGradsHost(i,j), tGradsGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tet10 jacobian at (0.25, 0.25, 0.25) for a cell that's
  in the reference configuration.  Jacobian should be identity.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, JacobianParentCoords )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    Plato::ScalarMultiVector tJacobianView("jacobian", Plato::Tet10::mNumSpatialDims, Plato::Tet10::mNumSpatialDims);
    Plato::ScalarArray3D tConfig("node locations", 1, Plato::Tet10::mNumNodesPerCell, Plato::Tet10::mNumSpatialDims);

    Kokkos::parallel_for("cell jacobian", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        tConfig(0,0,0) = 0.0; tConfig(0,0,1) = 0.0; tConfig(0,0,2) = 0.0;
        tConfig(0,1,0) = 1.0; tConfig(0,1,1) = 0.0; tConfig(0,1,2) = 0.0;
        tConfig(0,2,0) = 0.0; tConfig(0,2,1) = 1.0; tConfig(0,2,2) = 0.0;
        tConfig(0,3,0) = 0.0; tConfig(0,3,1) = 0.0; tConfig(0,3,2) = 1.0;
        tConfig(0,4,0) = 0.5; tConfig(0,4,1) = 0.0; tConfig(0,4,2) = 0.0;
        tConfig(0,5,0) = 0.5; tConfig(0,5,1) = 0.5; tConfig(0,5,2) = 0.0;
        tConfig(0,6,0) = 0.0; tConfig(0,6,1) = 0.5; tConfig(0,6,2) = 0.0;
        tConfig(0,7,0) = 0.0; tConfig(0,7,1) = 0.0; tConfig(0,7,2) = 0.5;
        tConfig(0,8,0) = 0.5; tConfig(0,8,1) = 0.0; tConfig(0,8,2) = 0.5;
        tConfig(0,9,0) = 0.0; tConfig(0,9,1) = 0.5; tConfig(0,9,2) = 0.5;

        Plato::Array<ElementType::mNumSpatialDims> tPoint;

        tPoint(0) = 0.25; tPoint(1) = 0.25; tPoint(2) = 0.25;
        auto tJacobian = ElementType::jacobian(tPoint, tConfig, ordinal);
        for(ordType i=0; i<Plato::Tet10::mNumSpatialDims; i++)
        {
            for(ordType j=0; j<Plato::Tet10::mNumSpatialDims; j++)
            {
                tJacobianView(i,j) = tJacobian(i,j);
            }
        }
    });

    auto tJacobianHost = Kokkos::create_mirror_view( tJacobianView );
    Kokkos::deep_copy( tJacobianHost, tJacobianView );

    std::vector<std::vector<Plato::Scalar>> tJacobianGold = { {1, 0, 0}, { 0, 1, 0 }, {0, 0, 1} };

    int tNumGold_I=tJacobianGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tJacobianGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tJacobianHost(i,j), tJacobianGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the Tri6 jacobian at (1/6, 1/6) for a cell that's
  in the reference configuration.  Jacobian should be identity.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tri6, JacobianParentCoords )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tri6>;

    Plato::ScalarMultiVector tJacobianView("jacobian", Plato::Tri6::mNumSpatialDims, Plato::Tri6::mNumSpatialDims);
    Plato::ScalarArray3D tConfig("node locations", 1, Plato::Tri6::mNumNodesPerCell, Plato::Tri6::mNumSpatialDims);

    Kokkos::parallel_for("cell jacobian", Kokkos::RangePolicy<int>(0,1), KOKKOS_LAMBDA(int ordinal)
    {
        tConfig(0,0,0) = 0.0; tConfig(0,0,1) = 0.0;
        tConfig(0,1,0) = 1.0; tConfig(0,1,1) = 0.0;
        tConfig(0,2,0) = 0.0; tConfig(0,2,1) = 1.0;
        tConfig(0,3,0) = 0.5; tConfig(0,3,1) = 0.0;
        tConfig(0,4,0) = 0.5; tConfig(0,4,1) = 0.5;
        tConfig(0,5,0) = 0.0; tConfig(0,5,1) = 0.5;

        Plato::Array<ElementType::mNumSpatialDims> tPoint;

        tPoint(0) = Plato::Scalar(1)/6; tPoint(1) = Plato::Scalar(1)/6;
        auto tJacobian = ElementType::jacobian(tPoint, tConfig, ordinal);
        for(ordType i=0; i<Plato::Tri6::mNumSpatialDims; i++)
        {
            for(ordType j=0; j<Plato::Tri6::mNumSpatialDims; j++)
            {
                tJacobianView(i,j) = tJacobian(i,j);
            }
        }
    });

    auto tJacobianHost = Kokkos::create_mirror_view( tJacobianView );
    Kokkos::deep_copy( tJacobianHost, tJacobianView );

    std::vector<std::vector<Plato::Scalar>> tJacobianGold = {{1, 0}, {0, 1}};

    int tNumGold_I=tJacobianGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        int tNumGold_J=tJacobianGold[0].size();
        for(int j=0; j<tNumGold_J; j++)
        {
            TEST_FLOATING_EQUALITY(tJacobianHost(i,j), tJacobianGold[i][j], 1e-13);
        }
    }
}

/******************************************************************************/
/*! 
  \brief Evaluate the surface area of a Tet10 face in the reference
         configuration.  Should evaluate to 1/2.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( Tet10, SurfaceArea )
{ 
    using ElementType = typename Plato::MechanicsElement<Plato::Tet10>;

    Plato::SurfaceArea<ElementType> surfaceArea;

    auto tCubatureWeights = ElementType::Face::getCubWeights();
    auto tCubaturePoints  = ElementType::Face::getCubPoints();
    auto tNumPoints = tCubatureWeights.size();

    constexpr auto tNodesPerFace = ElementType::mNumNodesPerFace;

    Plato::OrdinalVector tNodeOrds("node ordinals", Plato::Tet10::mNumNodesPerFace);

    Plato::ScalarMultiVector tJacobianView("jacobian", Plato::Tet10::mNumSpatialDims, Plato::Tet10::mNumSpatialDims);
    Plato::ScalarArray3D tConfig("node locations", 1, Plato::Tet10::mNumNodesPerCell, Plato::Tet10::mNumSpatialDims);

    Plato::ScalarVector tSurfaceArea("area at GP", tNumPoints);
    Kokkos::parallel_for("face area", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{1, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal, const Plato::OrdinalType & aPointOrdinal)

    {
        tConfig(0,0,0) = 0.0; tConfig(0,0,1) = 0.0; tConfig(0,0,2) = 0.0;
        tConfig(0,1,0) = 1.0; tConfig(0,1,1) = 0.0; tConfig(0,1,2) = 0.0;
        tConfig(0,2,0) = 0.0; tConfig(0,2,1) = 1.0; tConfig(0,2,2) = 0.0;
        tConfig(0,3,0) = 0.0; tConfig(0,3,1) = 0.0; tConfig(0,3,2) = 1.0;
        tConfig(0,4,0) = 0.5; tConfig(0,4,1) = 0.0; tConfig(0,4,2) = 0.0;
        tConfig(0,5,0) = 0.5; tConfig(0,5,1) = 0.5; tConfig(0,5,2) = 0.0;
        tConfig(0,6,0) = 0.0; tConfig(0,6,1) = 0.5; tConfig(0,6,2) = 0.0;
        tConfig(0,7,0) = 0.5; tConfig(0,7,1) = 0.0; tConfig(0,7,2) = 0.5;
        tConfig(0,8,0) = 0.0; tConfig(0,8,1) = 0.5; tConfig(0,8,2) = 0.5;
        tConfig(0,9,0) = 0.0; tConfig(0,9,1) = 0.0; tConfig(0,9,2) = 0.5;

        tNodeOrds(0) = 0; tNodeOrds(1) = 1; tNodeOrds(2) = 2;
        tNodeOrds(3) = 4; tNodeOrds(4) = 5; tNodeOrds(5) = 6;

        Plato::Array<Plato::Tet10::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
        for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<tNodesPerFace; tNodeOrd++)
        {
            tLocalNodeOrds(tNodeOrd) = tNodeOrds(aSideOrdinal*tNodesPerFace+tNodeOrd);
        }

        auto tCubatureWeight = tCubatureWeights(aPointOrdinal);
        auto tCubaturePoint = tCubaturePoints(aPointOrdinal);
        auto tBasisValues = ElementType::Face::basisValues(tCubaturePoint);
        auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

        Plato::Scalar tSurfaceAreaGP(0.0);
        surfaceArea(aSideOrdinal, tLocalNodeOrds, tBasisGrads, tConfig, tSurfaceAreaGP);

        tSurfaceArea(aPointOrdinal) = tSurfaceAreaGP*tCubatureWeight;

    });

    auto tAreasHost = Kokkos::create_mirror_view( tSurfaceArea );
    Kokkos::deep_copy( tAreasHost, tSurfaceArea );

    std::vector<Plato::Scalar> tAreasGold = {
        Plato::Scalar(1)/6, Plato::Scalar(1)/6, Plato::Scalar(1)/6
    };

    int tNumGold_I=tAreasGold.size();
    for(int i=0; i<tNumGold_I; i++)
    {
        TEST_FLOATING_EQUALITY(tAreasHost(i), tAreasGold[i], 1e-13);
    }

}
