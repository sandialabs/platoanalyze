#pragma once

#ifdef PLATO_UMFPACK

#include <cholmod.h>

#include <memory>
#include <optional>
#include <plato/utilities/StateCache.hpp>

#include "PlatoAbstractSolver.hpp"
#include "PlatoStaticsTypes.hpp"
#include "SuiteSparseUtils.hpp"

namespace Plato::alg
{
using RowMapVector = typename Plato::CrsMatrixType::RowMapVectorT;
using ColumnVector = typename Plato::CrsMatrixType::OrdinalVectorT;
using EntriesVector = typename Plato::CrsMatrixType::ScalarVectorT;

/// @brief RAII wrapper for setting up and destroying a `cholmod_common` object.
struct CholmodCommonSetupTeardown
{
    CholmodCommonSetupTeardown(Plato::LinearSystemType aLinearSystemType);
    ~CholmodCommonSetupTeardown();

    CholmodCommonSetupTeardown(const CholmodCommonSetupTeardown&) = delete;
    CholmodCommonSetupTeardown(CholmodCommonSetupTeardown&&) = delete;
    CholmodCommonSetupTeardown& operator=(const CholmodCommonSetupTeardown&) = delete;
    CholmodCommonSetupTeardown& operator=(CholmodCommonSetupTeardown&&) = delete;

    cholmod_common mValue;
};

/// @brief RAII wrapper for setting up and destroying a `cholmod_factor` object.
struct CholmodFactorSetupTeardown
{
    CholmodFactorSetupTeardown(cholmod_sparse* aCholmodSparse,
                               std::reference_wrapper<CholmodCommonSetupTeardown>&& aCholmodCommon);
    ~CholmodFactorSetupTeardown();

    CholmodFactorSetupTeardown(const CholmodFactorSetupTeardown&) = delete;
    CholmodFactorSetupTeardown(CholmodFactorSetupTeardown&&) noexcept;
    CholmodFactorSetupTeardown& operator=(const CholmodFactorSetupTeardown&) = delete;
    CholmodFactorSetupTeardown& operator=(CholmodFactorSetupTeardown&&) noexcept;

    std::reference_wrapper<CholmodCommonSetupTeardown> mCholmodCommon;
    cholmod_factor* mValue = nullptr;
};

/// @brief Interface for the cholmod solver. This only solves symmetric matrices, and only indefinite matrices if it is
/// constructed with SYMMETRIC_INDEFINITE.
class CHOLMODLinearSolver : public Plato::AbstractSolver
{
   public:
    CHOLMODLinearSolver(const Teuchos::ParameterList& aSolverParams,
                        Plato::LinearSystemType aLinearSystemType,
                        std::shared_ptr<Plato::MultipointConstraints> aMPCs = {});

    /// @brief Solves `Ax = b` with `A` given by @a aA, `b` by @a aB, and `x` stored in @a aX.
    ///
    /// @pre @a aA is symmetric and either positive definite, or if it is indefinite, this object was constructed with
    ///  SYMMETRIC_INDEFINITE. If @a aA has a non-symmetric pattern, an exception is thrown. If @a aA has a symmetric
    //   pattern but non-symmetric entries, the system is solved using only the lower diagonal.
    ///  If CHOLMODLinearSolver was constructed with SYMMETRIC_POSITIVE_DEFINITE, but @a aA is indefinite,
    ///  a cholmod may throw a floating point exception.
    void innerSolve(Plato::CrsMatrixType aA, Plato::ScalarVector aX, Plato::ScalarVector aB) override;

   private:
    using CholmodFactorCache =
        plato::utilities::StateCache<CholmodFactorSetupTeardown, const CSRMatrix&, cholmod_sparse*>;

    CholmodCommonSetupTeardown mCholmodCommon;
    CholmodFactorCache mCholmodFactorCache;
};

/// @brief Converts a CSRMatrix to a cholmod_sparse object in lower triangular form, and assumes that @a aMatrix is
/// symmetric.
[[nodiscard]] auto convertSymmetricCSRtoCHOLMODSparse(const CSRMatrix& aMatrix,
                                                      cholmod_common* const aCholmodCommon) -> cholmod_sparse*;

}  // namespace Plato::alg

#endif
