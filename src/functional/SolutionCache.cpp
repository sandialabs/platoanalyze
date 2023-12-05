#include "SolutionCache.hpp"
#include "PlatoStaticsTypes.hpp"
#include "Solutions.hpp"
#include "PlatoMesh.hpp"

namespace Plato::Functional
{

Plato::Solutions
SolutionCache::compute(const Plato::ScalarVector& aArg)
{
  std::size_t tDesignHash = mGenerateHash(aArg);
  if (tDesignHash != mDesignHash)
  {
    mSolution = mComputeSolution(aArg);
    mDesignHash = tDesignHash;
  }
  return mSolution;
}

}