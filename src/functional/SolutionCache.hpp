#include "PlatoStaticsTypes.hpp"
#include "Solutions.hpp"
#include "PlatoMesh.hpp"

#include <functional>

namespace Plato::Functional
{

class SolutionCache
{
public:
  using SolutionFunction = std::function<Plato::Solutions(const Plato::ScalarVector&)>;
  using HashingFunction = std::function<std::size_t(const Plato::ScalarVector&)>;

  SolutionCache
    (SolutionFunction aSF, 
     HashingFunction aHF) : 
       mComputeSolution(std::move(aSF)), 
       mGenerateHash(std::move(aHF)) 
  {}

  Plato::Solutions 
  compute(const Plato::ScalarVector& aArg);

private:
  SolutionFunction mComputeSolution;
  HashingFunction mGenerateHash;
  std::size_t mDesignHash;
  Plato::Solutions mSolution;

};

}