#include "AnalyzeCriterionInterface.hpp"

#include <array>
#include <plato/core/MeshProxy.hpp>
#include <string>
#include <string_view>

#include "FunctionalInterfaceUtilities.hpp"
#include "PlatoAbstractProblem.hpp"
#include "Solutions.hpp"
#include "alg/ParseInput.hpp"

namespace plato::functional
{
namespace
{
[[nodiscard]] Teuchos::ParameterList create_problem(const std::string_view aInputFileName)
{
    constexpr int tArgc = 2;
    std::string tAppName = "analyze";
    std::string tInputFileFlag = "--input-config=\"" + std::string{aInputFileName} + "\"";
    std::array<char*, 2> tArgv = {tAppName.data(), tInputFileFlag.data()};

    Plato::Comm::Machine tMachine = plato::functional::create_machine();
    return Plato::input_file_parsing(tArgc, tArgv.data(), tMachine);
}

[[nodiscard]] std::string first_criterion_name(const Teuchos::ParameterList& aProblem)
{
    auto tPlatoProblemList = aProblem.sublist("Plato Problem");
    assert(tPlatoProblemList.isSublist("Criteria"));
    auto tCriteriaList = tPlatoProblemList.sublist("Criteria");
    return tCriteriaList.name(tCriteriaList.begin());
}

[[nodiscard]] std::string first_file_name(const std::vector<std::string>& aFileNames)
{
    if (aFileNames.empty())
    {
        ANALYZE_THROWERR("An input file is required!");
    }
    return aFileNames.front();
}
}  // namespace

AnalyzeCriterionInterface::AnalyzeCriterionInterface(const std::vector<std::string>& aFileNames)
    : mFunctionalInterface(create_problem(first_file_name(aFileNames)))
{
}

double AnalyzeCriterionInterface::value(const core::MeshProxy& aMeshProxy) const
{
    const auto [tSolution, tControl] = mFunctionalInterface.solveProblem(aMeshProxy);

    const std::string tCriterionName = first_criterion_name(mFunctionalInterface.parameterList());
    const double tResult = mFunctionalInterface.problem().criterionValue(tControl, tSolution, tCriterionName);
    std::cout << "Criterion " << tCriterionName << " value: " << tResult << std::endl;
    return tResult;
}

std::vector<double> AnalyzeCriterionInterface::gradient(const core::MeshProxy& aMeshProxy) const
{
    const auto [tSolution, tControl] = mFunctionalInterface.solveProblem(aMeshProxy);

    const std::string tCriterionName = first_criterion_name(mFunctionalInterface.parameterList());
    const Plato::ScalarVector tGradient =
        aMeshProxy.mNodalDensities.empty()
            ? mFunctionalInterface.problem().criterionGradientX(tControl, tSolution, tCriterionName)
            : mFunctionalInterface.problem().criterionGradient(tControl, tSolution, tCriterionName);

    return plato::functional::to_std_vector(tGradient);
}
}  // namespace plato::functional

namespace plato
{
std::unique_ptr<criteria::library::CriterionInterface> plato_create_criterion(
    const std::vector<std::string>& aFileNames)
{
    return std::make_unique<plato::functional::AnalyzeCriterionInterface>(aFileNames);
}
}  // namespace plato
