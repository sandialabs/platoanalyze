#ifndef PLATO_FUNCTIONAL_ANALYZECRITERIONINTERFACE_H
#define PLATO_FUNCTIONAL_ANALYZECRITERIONINTERFACE_H

#include "CriterionInterface.hpp"
#include "FunctionalInterface.hpp"

namespace Plato::Functional
{
/// @brief Main criterion interface to PlatoFunctional.
///
/// Given an input file, this will compute the first criterion listed in
/// the value member function and its gradient in the gradient member.
class AnalyzeCriterionInterface : public Plato::Functional::CriterionInterface
{
public:

  ///@brief Construct a new Analyze Criterion Interface object
  ///
  ///@param aFileNames 
  explicit AnalyzeCriterionInterface(const std::vector<std::string>& aFileNames);

  ///@brief Return the value of the criterion at the control specified by the MeshProxy
  ///
  ///@param aMeshProxy 
  ///@return double 
  [[nodiscard]]
  double value(const MeshProxy& aMeshProxy) const override;

  ///@brief Return the gradient of the criterion evaluated at the control specified by the MeshProxy 
  ///
  ///@param aMeshProxy 
  ///@return std::vector<double> 
  [[nodiscard]]
  std::vector<double> gradient(const MeshProxy& aMeshProxy) const override;
private:
  mutable FunctionalInterface mFunctionalInterface;
};
}

#endif
