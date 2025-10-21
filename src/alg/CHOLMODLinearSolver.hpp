#pragma once

#ifdef PLATO_UMFPACK

#include <cholmod.h>

#include <filesystem>
#include <functional>
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
struct CHOLMODCommonSetupTeardown
{
    CHOLMODCommonSetupTeardown(Plato::LinearSystemType aLinearSystemType);
    ~CHOLMODCommonSetupTeardown();

    CHOLMODCommonSetupTeardown(const CHOLMODCommonSetupTeardown &) = delete;
    CHOLMODCommonSetupTeardown(CHOLMODCommonSetupTeardown &&) = delete;
    CHOLMODCommonSetupTeardown &operator=(const CHOLMODCommonSetupTeardown &) = delete;
    CHOLMODCommonSetupTeardown &operator=(CHOLMODCommonSetupTeardown &&) = delete;

    cholmod_common mValue;
};

/// @brief A generic RAII wrapper for a pointer to a CHOLMOD object that is freed on destruction.
template <typename CHOLMODObject>
class CHOLMODObjectWrapper
{
   public:
    /// @brief On destruction @a aDeleter is executed with @a aObject passed as an argument. @a aDeleter should call the
    /// appropriate cholmod free function to deallocate the object.
    template <typename Deleter>
    CHOLMODObjectWrapper(CHOLMODObject *const aObject,
                         std::reference_wrapper<CHOLMODCommonSetupTeardown> &&aCHOLMODCommon,
                         Deleter &&aDeleter);

    CHOLMODObjectWrapper(const CHOLMODObjectWrapper &) = delete;
    CHOLMODObjectWrapper(CHOLMODObjectWrapper &&) noexcept;
    CHOLMODObjectWrapper &operator=(const CHOLMODObjectWrapper &) = delete;
    CHOLMODObjectWrapper &operator=(CHOLMODObjectWrapper &&) noexcept;
    ~CHOLMODObjectWrapper();

    CHOLMODObject *mObject = nullptr;

   private:
    std::reference_wrapper<CHOLMODCommonSetupTeardown> mCHOLMODCommon;
    std::function<void(CHOLMODObject **, cholmod_common *)> mDeleter;
};

/// @brief Interface for the cholmod solver. This only solves symmetric matrices, and only indefinite matrices if it is
/// constructed with SYMMETRIC_INDEFINITE.
class CHOLMODLinearSolver : public Plato::AbstractSolver
{
   public:
    CHOLMODLinearSolver(const Teuchos::ParameterList &aSolverParams,
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
    using CHOLMODFactorCache = plato::utilities::StateCache<CHOLMODObjectWrapper<cholmod_factor>,
                                                            const CrsRowsColumnsValues<Plato::OrdinalType> &,
                                                            cholmod_sparse *>;

    CHOLMODCommonSetupTeardown mCHOLMODCommon;
    CHOLMODFactorCache mCHOLMODFactorCache;
};

/// @brief Converts a CRSMatrix to a cholmod_sparse object in lower triangular form, and assumes that @a aMatrix is
/// symmetric.
[[nodiscard]] auto symmetric_CRS_to_CHOLMOD_sparse(const CrsRowsColumnsValues<Plato::OrdinalType> &aMatrix,
                                                   CHOLMODCommonSetupTeardown &aCHOLMODCommon)
    -> CHOLMODObjectWrapper<cholmod_sparse>;

/// @brief Returns the filename used to print unsolvable matrices to from CHOLMOD.
[[nodiscard]] auto bad_cholmod_matrix_file_path() -> std::filesystem::path;

template <typename CHOLMODObject>
template <typename Deleter>
CHOLMODObjectWrapper<CHOLMODObject>::CHOLMODObjectWrapper(
    CHOLMODObject *const aObject,
    std::reference_wrapper<CHOLMODCommonSetupTeardown> &&aCHOLMODCommon,
    Deleter &&aDeleter)
    : mObject{aObject}, mCHOLMODCommon{std::move(aCHOLMODCommon)}, mDeleter{std::forward<Deleter>(aDeleter)}
{
}

template <typename CHOLMODObject>
CHOLMODObjectWrapper<CHOLMODObject>::~CHOLMODObjectWrapper()
{
    if (mObject)
    {
        mDeleter(&mObject, &mCHOLMODCommon.get().mValue);
    }
}

template <typename CHOLMODObject>
CHOLMODObjectWrapper<CHOLMODObject>::CHOLMODObjectWrapper(CHOLMODObjectWrapper &&aOther) noexcept
    : mObject{aOther.mObject}, mCHOLMODCommon{aOther.mCHOLMODCommon}, mDeleter{std::move(aOther.mDeleter)}
{
    aOther.mObject = nullptr;
}

template <typename CHOLMODObject>
auto CHOLMODObjectWrapper<CHOLMODObject>::operator=(CHOLMODObjectWrapper &&aOther) noexcept -> CHOLMODObjectWrapper &
{
    if (this != &aOther)
    {
        std::swap(aOther.mObject, mObject);
        std::swap(aOther.mCHOLMODCommon, mCHOLMODCommon);
        std::swap(aOther.mDeleter, mDeleter);
    }
    return *this;
}

}  // namespace Plato::alg

#endif
