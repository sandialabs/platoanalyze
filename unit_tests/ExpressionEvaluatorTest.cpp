#include "util/PlatoTestHelpers.hpp"
#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "ParseTools.hpp"
#include "PlatoMathTypes.hpp"

#include "ExpressionEvaluator.hpp"

namespace {
  constexpr double SPEED_OF_LIGHT = 299792458.0;
}

namespace PlatoUnitTests
{
/******************************************************************************/
/*!
  \brief Unit tests for Plato::ExpressionEvaluator

  Test of a ScalarVectorT (1D) that is indexed based on the cell ordinal.

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionEvaluator_1D_index)
{
    constexpr Plato::OrdinalType tNumCells = 10;
    constexpr Plato::OrdinalType tNumPoints = 1;

    Plato::ScalarMultiVectorT<Plato::Scalar> tEnergy("energy", tNumCells, tNumPoints);
    Plato::ScalarVectorT<Plato::Scalar> tMass("mass", tNumCells);

    Kokkos::parallel_for("Set the mass for each Element",
    Kokkos::RangePolicy<>(0,tNumCells),
    KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
      tMass(aCellOrdinal) = aCellOrdinal;
    });

    Plato::ExpressionEvaluator<
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarVectorT<Plato::Scalar>,
        Plato::Scalar > tExpEval;

    tExpEval.parse_expression("mass*c^2");
    tExpEval.setup_storage(tNumCells, tNumPoints);

    tExpEval.set_variable("c", SPEED_OF_LIGHT);

    // Test of a ScalarVectorT (1D) that is indexed based on the cell
    // ordinal.  That is because the result array, energy is N x 1.
    // That is because the mass is a 1D array it will be indexed by
    // the cell (first) ordinal WITHIN the expresion evaluator.
    tExpEval.set_variable("mass", tMass);

    tExpEval.evaluate_expression( tEnergy );

    Kokkos::fence();

    auto tEnergy_Host = Kokkos::create_mirror_view(tEnergy);
    Kokkos::deep_copy(tEnergy_Host, tEnergy);

    for(int i=0; i<tNumCells; ++i)
    {
      for(int j=0; j<tNumPoints; ++j)
      {
        TEST_ASSERT(tEnergy_Host(i,j) == i * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
      }
    }

    tExpEval.clear_storage();
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::ExpressionEvaluator

  Test of a ScalarVectorT (1D) that is set on per cell ordinal basis.

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionEvaluator_1D_const)
{
    constexpr Plato::OrdinalType tNumCells = 10;
    constexpr Plato::OrdinalType tNumPoints = 1;

    Plato::ScalarMultiVectorT<Plato::Scalar> tEnergy("energy", tNumCells, tNumPoints);
    Plato::ScalarVectorT<Plato::Scalar> tMass("mass", tNumCells);

    Kokkos::parallel_for("Set the mass for each Element",
    Kokkos::RangePolicy<>(0,tNumCells),
    KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
      tMass(aCellOrdinal) = aCellOrdinal;
    });

    Plato::ExpressionEvaluator<
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarVectorT<Plato::Scalar>,
        Plato::Scalar > tExpEval;

    tExpEval.parse_expression("mass*c^2");
    tExpEval.setup_storage(tNumCells, tNumPoints);

    tExpEval.set_variable("c", SPEED_OF_LIGHT);

#if defined(EXPRESSION_EVALUATOR_USE_PER_THREAD_VAR_ASSIGNMENT)
    Kokkos::parallel_for("Compute the energy for each Element",
    Kokkos::RangePolicy<>(0,tNumCells),
    KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
      // Test of a ScalarVectorT (1D) that is set on per cell
      // ordinal basis. That is the mass is set on a cell by cell
      // basis as a constant.

      // NOTE: Indexing the mass is for testing only and should
      // NEVER be used in practice. The previous test demonstrates
      // the correct practice.
      tExpEval.set_variable("mass", tMass(aCellOrdinal), aCellOrdinal);

      tExpEval.evaluate_expression( aCellOrdinal, tEnergy );
    });
#else
    // Test of a ScalarVectorT (1D) that is indexed based on the cell
    // ordinal.  That is because the result array, energy is N x 1.
    // That is because the mass is a 1D array it will be indexed by
    // the cell (first) ordinal WITHIN the expresion evaluator.
    tExpEval.set_variable("mass", tMass);
    tExpEval.evaluate_expression( tEnergy );
#endif

    Kokkos::fence();

    auto tEnergy_Host = Kokkos::create_mirror_view(tEnergy);
    Kokkos::deep_copy(tEnergy_Host, tEnergy);

    for(int i=0; i<tNumCells; ++i)
    {
      for(int j=0; j<tNumPoints; ++j)
      {
        TEST_ASSERT(tEnergy_Host(i,j) == i * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
      }
    }

    tExpEval.clear_storage();
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::ExpressionEvaluator

  Test of a ScalarMultiVectorT (2D) that is indexed based on the cell ordinal.

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionEvaluator_2D_index)
{
    constexpr Plato::OrdinalType tNumCells  = 10;
    constexpr Plato::OrdinalType tNumPoints =  5;
    Plato::ScalarMultiVectorT<Plato::Scalar> tEnergy("energy", tNumCells, tNumPoints);
    Plato::ScalarMultiVectorT<Plato::Scalar> tMass("mass", tNumCells, tNumPoints);
    Kokkos::parallel_for("Set the mass for each Element",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType aCellOrdinal,
                  const Plato::OrdinalType aPointOrdinal)
    {
      tMass(aCellOrdinal, aPointOrdinal) = aCellOrdinal * aPointOrdinal;
    });

    Plato::ExpressionEvaluator<
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarVectorT<Plato::Scalar>,
        Plato::Scalar > tExpEval;

    tExpEval.parse_expression("mass*c^2");
    tExpEval.setup_storage(tNumCells, tNumPoints);

    tExpEval.set_variable("c", SPEED_OF_LIGHT);

    // Test of a ScalarMultiVectorT (2D) that is indexed based on the
    // cell and point ordinal. That is because the mass is a 2D array
    // it will be index by the cell and point ordinal WITHIN the
    // expresion evaluator.
    tExpEval.set_variable("mass", tMass);

    tExpEval.evaluate_expression( tEnergy );

    Kokkos::fence();

    auto tEnergy_Host = Kokkos::create_mirror_view(tEnergy);
    Kokkos::deep_copy(tEnergy_Host, tEnergy);

    for(int i=0; i<tNumCells; ++i)
    {
      for(int j=0; j<tNumPoints; ++j)
      {
        TEST_ASSERT(tEnergy_Host(i,j) == i*j * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
      }
    }

    tExpEval.clear_storage();
}


/******************************************************************************/
/*!
  \brief Unit tests for Plato::ExpressionEvaluator

  Test of a ScalarVectorT (1D) that is set on per cell ordinal basis.

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionEvaluator_2D_const)
{
    constexpr Plato::OrdinalType tNumCells  = 10;
    constexpr Plato::OrdinalType tNumPoints =  5;
    Plato::ScalarMultiVectorT<Plato::Scalar> tEnergy("energy", tNumCells, tNumPoints);
    Plato::ScalarMultiVectorT<Plato::Scalar> tMass("mass", tNumCells, tNumPoints);
    Kokkos::parallel_for("Set the mass for each Element",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType aCellOrdinal, const Plato::OrdinalType aPointOrdinal)
    {
      tMass(aCellOrdinal, aPointOrdinal) = aCellOrdinal * aPointOrdinal;
    });

    Plato::ExpressionEvaluator<
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarVectorT<Plato::Scalar>,
        Plato::Scalar > tExpEval;

    tExpEval.parse_expression("mass*c^2");
    tExpEval.setup_storage(tNumCells, tNumPoints);

    tExpEval.set_variable("c", SPEED_OF_LIGHT);

#if defined(EXPRESSION_EVALUATOR_USE_PER_THREAD_VAR_ASSIGNMENT)
    Kokkos::parallel_for("Compute the energy for each Element",
    Kokkos::RangePolicy<>(0,tNumCells),
    KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
      // Test of a ScalarVectorT (1D) that is set on per cell
      // ordinal basis and indexed based on the point ordinal. That
      // is the mass is set on a cell by cell basis and because the
      // result is a 2D array and the mass as a 1d array will be
      // indexed by the point (second) ordinal WITHIN the expresion
      // evaluator.

      // Contrast to the first test. When the result is a 1D array,
      // 1D variables are indexed based on the cell (first) ordinal.
      // When the result is a 2D array, 1D variables are indexed
      // based on the point (second) ordinal.

      // NOTE: Taking the subview is for testing only and should
      // NEVER be used in practice. The previous test demonstrates
      // the correct practice.
      auto tMass_sub = subview (tMass, aCellOrdinal, Kokkos::ALL());

      tExpEval.set_variable("mass", tMass_sub, aCellOrdinal);

      tExpEval.evaluate_expression( aCellOrdinal, tEnergy );
    })
#else
    // Test of a ScalarMultiVectorT (2D) that is indexed based on the
    // cell and point ordinal. That is because the mass is a 2D array
    // it will be index by the cell and point ordinal WITHIN the
    // expresion evaluator.
    tExpEval.set_variable("mass", tMass);
    tExpEval.evaluate_expression( tEnergy );
#endif

    Kokkos::fence();

    auto tEnergy_Host = Kokkos::create_mirror_view(tEnergy);
    Kokkos::deep_copy(tEnergy_Host, tEnergy);

    for(int i=0; i<tNumCells; ++i)
    {
      for(int j=0; j<tNumPoints; ++j)
      {
        TEST_ASSERT(tEnergy_Host(i,j) == i*j * SPEED_OF_LIGHT * SPEED_OF_LIGHT);
      }
    }

    tExpEval.clear_storage();
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::ExpressionEvaluator

  Memory test of a ScalarMultiVectorT (2D) that is indexed based on
  the cell ordinal.

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionEvaluator_2D_index_memory)
{
  // Do 100 loops with 1000 x 1000 cells to test the memory.
  for( size_t t=0; t<100; ++t)
  {
    constexpr Plato::OrdinalType tNumCells  = 1000;
    constexpr Plato::OrdinalType tNumPoints = 1000;
    Plato::ScalarMultiVectorT<Plato::Scalar> tEnergy("energy", tNumCells, tNumPoints);
    Plato::ScalarMultiVectorT<Plato::Scalar> tMass("mass", tNumCells, tNumPoints);
    Kokkos::parallel_for("Set the mass for each Element",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType aCellOrdinal,
                  const Plato::OrdinalType aPointOrdinal)
    {
      tMass(aCellOrdinal, aPointOrdinal) = aCellOrdinal * aPointOrdinal;
    });

    Plato::ExpressionEvaluator<
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarVectorT<Plato::Scalar>,
        Plato::Scalar > tExpEval;

    tExpEval.parse_expression("mass");
    tExpEval.setup_storage(tNumCells, tNumPoints);

    // Test of a ScalarMultiVectorT (2D) that is indexed based on the
    // cell and point ordinal. That is because the mass is a 2D array
    // it will be index by the cell and point ordinal WITHIN the
    // expresion evaluator.
    tExpEval.set_variable("mass", tMass);

    tExpEval.evaluate_expression( tEnergy );

    Kokkos::fence();

    auto tEnergy_Host = Kokkos::create_mirror_view(tEnergy);
    Kokkos::deep_copy(tEnergy_Host, tEnergy);

    for(int i=0; i<tNumCells; ++i)
    {
      for(int j=0; j<tNumPoints; ++j)
      {
        TEST_ASSERT(tEnergy_Host(i,j) == static_cast<double>(i*j) );
      }
    }

    tExpEval.clear_storage();
  }
}

/******************************************************************************/
/*!
  \brief Unit tests for Plato::ExpressionEvaluator

  Test of a ScalarVectorT (1D) that is set on per cell ordinal basis.

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ExpressionEvaluator_2D_const_memory)
{
  // Do 100 loops with 1000 x 1000 cells to test the memory.
  for( size_t t=0; t<100; ++t)
  {
    constexpr Plato::OrdinalType tNumCells  = 1000;
    constexpr Plato::OrdinalType tNumPoints = 1000;
    Plato::ScalarMultiVectorT<Plato::Scalar> tEnergy("energy", tNumCells, tNumPoints);
    Plato::ScalarMultiVectorT<Plato::Scalar> tMass("mass", tNumCells, tNumPoints);
    Kokkos::parallel_for("Set the mass for each Element",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const Plato::OrdinalType aCellOrdinal, const Plato::OrdinalType aPointOrdinal)
    {
        tMass(aCellOrdinal, aPointOrdinal) = aCellOrdinal * aPointOrdinal;
    });

    Plato::ExpressionEvaluator<
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarMultiVectorT<Plato::Scalar>,
        Plato::ScalarVectorT<Plato::Scalar>,
        Plato::Scalar > tExpEval;

    tExpEval.parse_expression("mass");
    tExpEval.setup_storage(tNumCells, tNumPoints);

#if !defined(EXPRESSION_EVALUATOR_USE_PER_THREAD_VAR_ASSIGNMENT)
    // Test of a ScalarMultiVectorT (2D) that is indexed based on the
    // cell and point ordinal. That is because the mass is a 2D array
    // it will be index by the cell and point ordinal WITHIN the
    // expresion evaluator.
    tExpEval.set_variable("mass", tMass);
#endif

    Kokkos::parallel_for("Compute the energy for each Element",
    Kokkos::RangePolicy<>(0,tNumCells),
    KOKKOS_LAMBDA(Plato::OrdinalType aCellOrdinal)
    {
#if defined(EXPRESSION_EVALUATOR_USE_PER_THREAD_VAR_ASSIGNMENT)
        // Test of a ScalarVectorT (1D) that is set on per cell
        // ordinal basis and indexed based on the point ordinal. That
        // is the mass is set on a cell by cell basis and because the
        // result is a 2D array and the mass as a 1d array will be
        // indexed by the point (second) ordinal WITHIN the expresion
        // evaluator.

        // Contrast to the first test. When the result is a 1D array,
        // 1D variables are indexed based on the cell (first) ordinal.
        // When the result is a 2D array, 1D variables are indexed
        // based on the point (second) ordinal.

        // NOTE: Taking the subview is for testing only and should
        // NEVER be used in practice. The previous test demonstrates
        // the correct practice.
        auto tMass_sub = subview (tMass, aCellOrdinal, Kokkos::ALL());

        tExpEval.set_variable("mass", tMass_sub, aCellOrdinal);
#endif
        tExpEval.evaluate_expression( aCellOrdinal, tEnergy );
    });

    Kokkos::fence();

    auto tEnergy_Host = Kokkos::create_mirror_view(tEnergy);
    Kokkos::deep_copy(tEnergy_Host, tEnergy);

    for(int i=0; i<tNumCells; ++i)
    {
      for(int j=0; j<tNumPoints; ++j)
      {
        TEST_ASSERT(tEnergy_Host(i,j) == static_cast<double>(i*j) );
      }
    }

    tExpEval.clear_storage();
  }
}

}
