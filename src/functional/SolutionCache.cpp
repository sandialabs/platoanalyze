#include "SolutionCache.hpp"

#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "Solutions.hpp"

namespace plato::functional
{

Plato::Solutions SolutionCache::compute(const Plato::ScalarVector& aArg)
{
    std::size_t tDesignHash = mGenerateHash(aArg);
    if (tDesignHash != mDesignHash)
    {
        mSolution = mComputeSolution(aArg);
        mDesignHash = tDesignHash;
    }
    return mSolution;
}

}  // namespace plato::functional