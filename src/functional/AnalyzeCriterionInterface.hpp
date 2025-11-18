#ifndef PLATO_FUNCTIONAL_ANALYZECRITERIONINTERFACE_H
#define PLATO_FUNCTIONAL_ANALYZECRITERIONINTERFACE_H

#include <plato/criteria/library/CriterionInterface.hpp>

#include "functional/FunctionalInterface.hpp"

namespace plato::functional
{
/// @brief Main criterion interface to PlatoFunctional.
///
/// Given an input file, this will compute the first criterion listed in
/// the value member function and its gradient in the gradient member.
class AnalyzeCriterionInterface : public plato::criteria::library::CriterionInterface
{
   public:
    ///@brief Construct a new Analyze Criterion Interface object
    explicit AnalyzeCriterionInterface(const std::vector<std::string>& aFileNames);

    ///@brief Return the value of the criterion at the control specified by @a aAnalysisDomainMesh
    [[nodiscard]] double value(const analysis::AnalysisDomainMesh& aAnalysisDomainMesh) const override;

    ///@brief Return the gradient of the criterion evaluated at the control specified by @a aAnalysisDomainMesh
    [[nodiscard]] std::vector<double> gradient(const analysis::AnalysisDomainMesh& aAnalysisDomainMesh) const override;

   private:
    mutable FunctionalInterface mFunctionalInterface;
};
}  // namespace plato::functional

extern "C" std::unique_ptr<::plato::criteria::library::CriterionInterface> plato_create_criterion(
    const std::vector<std::string>& aFileNames);

#endif
