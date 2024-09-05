#ifndef PLATO_FUNCTIONAL_ANALYZECRITERIONINTERFACE_H
#define PLATO_FUNCTIONAL_ANALYZECRITERIONINTERFACE_H

#include <plato/criteria/CriterionInterface.hpp>

#include "FunctionalInterface.hpp"

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

    ///@brief Return the value of the criterion at the control specified by @a aMeshDesignVariables
    [[nodiscard]] double value(const design_variables::MeshDesignVariables& aMeshDesignVariables) const override;

    ///@brief Return the gradient of the criterion evaluated at the control specified by @a aMeshDesignVariables
    [[nodiscard]] std::vector<double> gradient(
        const design_variables::MeshDesignVariables& aMeshDesignVariables) const override;

   private:
    mutable FunctionalInterface mFunctionalInterface;
};
}  // namespace plato::functional

extern "C" std::unique_ptr<::plato::criteria::library::CriterionInterface> plato_create_criterion(
    const std::vector<std::string>& aFileNames);

#endif
