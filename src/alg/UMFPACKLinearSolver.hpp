#pragma once

#ifdef PLATO_UMFPACK

#include <filesystem>
#include <plato/utilities/StateCache.hpp>
#include <string>
#include <vector>

#include "PlatoAbstractSolver.hpp"
#include "PlatoStaticsTypes.hpp"
#include "alg/SuiteSparseUtils.hpp"

namespace Plato::alg
{
/// @brief Used for custom deleter of UMFPACK symbolic object.
struct UMFPACKSymbolicDeleter
{
    void operator()(void* aUMFPACKSymbolic);
};

/// @brief Interface to the UMFPACK sparse direct linear solver. May be used for any type of sparse system.
///
/// For symmetric matrices, see CHOLMODLinearSolver.
class UMFPACKLinearSolver : public Plato::AbstractSolver
{
   public:
    UMFPACKLinearSolver(const Teuchos::ParameterList& aSolverParams,
                        std::shared_ptr<Plato::MultipointConstraints> aMPCs = nullptr);

    void innerSolve(Plato::CrsMatrix<Plato::OrdinalType> aCSRMatrix,
                    Plato::ScalarVector aX,
                    Plato::ScalarVector aB) override;

   private:
    using UMFPACKSymbolic = std::unique_ptr<void, UMFPACKSymbolicDeleter>;
    using UMFPACKSymbolicCache = plato::utilities::StateCache<UMFPACKSymbolic, const CSCMatrix&, const CrsMatrixType&>;

    UMFPACKSymbolicCache mUMFPACKSymbolicCache;
};

/// @brief Returns the filename used to print unsolvable matrices to from UMFPACK.
[[nodiscard]] auto bad_umfpack_matrix_file_path() -> std::filesystem::path;

}  // namespace Plato::alg

#endif
