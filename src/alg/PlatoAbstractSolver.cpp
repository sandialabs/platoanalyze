#include "PlatoAbstractSolver.hpp"

#include "PlatoStaticsTypes.hpp"
#include "MultipointConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "BLAS1.hpp"

#include <string_view>

namespace Plato {

namespace
{
constexpr std::string_view kRelativeDiagonalOffsetName = "Relative Diagonal Offset";
constexpr Plato::Scalar kDefaultDiagonalOffset = 0.0;

Plato::Scalar parseDiagonalOffset(const Teuchos::ParameterList & aSolverParams)
{
  if(aSolverParams.isType<Plato::Scalar>(kRelativeDiagonalOffsetName.data()))
  {
    const auto tOffset = aSolverParams.get<Plato::Scalar>(kRelativeDiagonalOffsetName.data());
    if(tOffset < 0.0)
    {
      ANALYZE_THROWERR("Linear solver settings: Relative Diagonal Offset cannot be less than 0.0.");
    }
    return tOffset;
  }
  else
  {
    return kDefaultDiagonalOffset;
  }
}
}

AbstractSolver::AbstractSolver(const Teuchos::ParameterList & aSolverParams) :
  mAlpha(parseDiagonalOffset(aSolverParams))
{
}

AbstractSolver::AbstractSolver(
  const Teuchos::ParameterList & aSolverParams,
  std::shared_ptr<Plato::MultipointConstraints> aMPCs) :
  mSystemMPCs(aMPCs),
  mAlpha(parseDiagonalOffset(aSolverParams))
{}

void AbstractSolver::solve(Plato::CrsMatrix<int> aAf, Plato::ScalarVector aX,
                           Plato::ScalarVector aB, bool aAdjointFlag) {
  
  Plato::Scalar tOffset;
  if (mAlpha != 0.0) {
    tOffset = mAlpha*diagonalAveAbs(aAf);
    shiftDiagonal(aAf, tOffset);
  }

  if (mSystemMPCs) {

    Teuchos::RCP<Plato::CrsMatrixType> aA(&aAf, /*hasOwnership=*/false);

    const Plato::OrdinalType tNumNodes = mSystemMPCs->getNumTotalNodes();
    const Plato::OrdinalType tNumCondensedNodes =
        mSystemMPCs->getNumCondensedNodes();

    Plato::OrdinalType tNumDofsPerNode = mSystemMPCs->getNumDofsPerNode();
    auto tNumDofs = tNumNodes * tNumDofsPerNode;
    auto tNumCondensedDofs = tNumCondensedNodes * tNumDofsPerNode;

    // get MPC condensation matrices and RHS
    Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrix =
        mSystemMPCs->getTransformMatrix();
    Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrixTranspose =
        mSystemMPCs->getTransformMatrixTranspose();

    Plato::ScalarVector tMpcRhs;
    if (aAdjointFlag == true) {
      Kokkos::resize(tMpcRhs, tNumDofs);
      Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tMpcRhs);
    } else {
      tMpcRhs = mSystemMPCs->getRhsVector();
    }

    // build condensed matrix
    auto tCondensedALeft = Teuchos::rcp(new Plato::CrsMatrixType(
        tNumDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode));
    auto tCondensedA = Teuchos::rcp(
        new Plato::CrsMatrixType(tNumCondensedDofs, tNumCondensedDofs,
                                 tNumDofsPerNode, tNumDofsPerNode));

    Plato::MatrixMatrixMultiply(aA, tTransformMatrix, tCondensedALeft);
    Plato::MatrixMatrixMultiply(tTransformMatrixTranspose, tCondensedALeft,
                                tCondensedA);

    // build condensed vector
    Plato::ScalarVector tInnerB = aB;
    Plato::blas1::scale(-1.0, tMpcRhs);
    Plato::MatrixTimesVectorPlusVector(aA, tMpcRhs, tInnerB);

    Plato::ScalarVector tCondensedB("Condensed RHS Vector", tNumCondensedDofs);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tCondensedB);

    Plato::MatrixTimesVectorPlusVector(tTransformMatrixTranspose, tInnerB,
                                       tCondensedB);

    // solve condensed system
    Plato::ScalarVector tCondensedX("Condensed Solution", tNumCondensedDofs);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tCondensedX);

    this->innerSolve(*tCondensedA, tCondensedX, tCondensedB);

    // get full solution vector
    Plato::ScalarVector tFullX("Full State Solution", aX.extent(0));
    Plato::blas1::copy(tMpcRhs, tFullX);
    Plato::blas1::scale(-1.0, tFullX); // since tMpcRhs was scaled by -1 above,
                                       // set back to original values

    Plato::MatrixTimesVectorPlusVector(tTransformMatrix, tCondensedX, tFullX);
    Plato::blas1::axpy<Plato::ScalarVector>(1.0, tFullX, aX);
  } else {
    this->innerSolve(aAf, aX, aB);
  }

  if (mAlpha) {
    shiftDiagonal(aAf, -tOffset);
  }
}

} // namespace Plato
