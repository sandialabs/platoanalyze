#ifndef PLATO_PROBLEM_PARABOLIC_PARSINGUTILITIES_DEF
#define PLATO_PROBLEM_PARABOLIC_PARSINGUTILITIES_DEF

#include <Teuchos_ParameterList.hpp>
#include <map>
#include <memory>
#include <string>

#include "domain/SpatialModel.hpp"
#include "mesh/ComputedField.hpp"
#include "parsing/TeuchosParsingUtilities.hpp"
#include "problem/parabolic/ScalarFunctionBaseFactory.hpp"
#include "solver/PlatoAbstractSolver.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace Plato
{
class MultipointConstraints;
class AbstractSolver;

namespace Comm
{
struct Machine;
}
}  // namespace Plato

namespace plato::algorithms::nonlinear_solvers
{
class NewtonSolver;
}

namespace plato::parabolic
{
/// @brief returns a std::map of criteria defined in @a aProblemParams, constructing them using @a aSpatialModel, and @a
/// aDataMap
template <typename PhysicsType, typename CriterionType>
auto parse_criteria(Teuchos::ParameterList& aProblemParams,
                    const domain::SpatialModel& aSpatialModel,
                    Plato::DataMap& aDataMap) -> std::map<std::string, CriterionType>;

/// @brief returns a shared_ptr of MultiPointConstraints from MPCs defined in @a aProblemParams.
/// If no MPCs are defined, returns a nullptr.
auto parse_multipoint_constraints(Teuchos::ParameterList& aProblemParams,
                                  const domain::SpatialModel& aSpatialModel,
                                  const Plato::OrdinalType aNumDofsPerNode)
    -> std::shared_ptr<Plato::MultipointConstraints>;

/// @brief returns a Plato::rcp to an AbstractLinearSolver constructed from @a aProblemParams, with linear system type
/// determined from @a aPhysics. If no MPCs are defined, returns a nullptr.
auto parse_linear_solver(Teuchos::ParameterList& aSolverParameters,
                         const std::string& aPhysics,
                         const Plato::OrdinalType aNumNodes,
                         Plato::Comm::Machine aMachine,
                         const Plato::OrdinalType aNumDofsPerNode,
                         const std::shared_ptr<Plato::MultipointConstraints>& aMPCs)
    -> Plato::rcp<Plato::AbstractSolver>;

/// @brief returns a NewtonSolver constructed from @a aProblemParams and @a aLinearSolver
auto parse_newton_solver(Teuchos::ParameterList& aProblemParams, const Plato::rcp<Plato::AbstractSolver>& aLinearSolver)
    -> algorithms::nonlinear_solvers::NewtonSolver;

/// @brief populates the ScalarVector @a aInitialState with initial state values defined in @a aProblemParams
template <typename ElementType>
void parse_initial_state(Teuchos::ParameterList& aProblemParams,
                         Plato::ScalarVector& aInitialState,
                         Plato::Mesh aMesh,
                         const std::vector<std::string>& aDofNames);

template <typename PhysicsType, typename CriterionType>
auto parse_criteria(Teuchos::ParameterList& aProblemParams,
                    const domain::SpatialModel& aSpatialModel,
                    Plato::DataMap& aDataMap) -> std::map<std::string, CriterionType>
{
    if (aProblemParams.isSublist("Criteria"))
    {
        std::map<std::string, CriterionType> tCriterionMap;
        if (aProblemParams.isSublist("Criteria"))
        {
            auto tAddCriteria = [&tCriterionMap, &aSpatialModel, &aDataMap, &aProblemParams,
                                 tCriterionBaseFactory = Plato::Parabolic::ScalarFunctionBaseFactory<PhysicsType>{}](
                                    const std::string& aName)
            {
                const auto tCriterion = tCriterionBaseFactory.create(aSpatialModel, aDataMap, aProblemParams, aName);
                if (tCriterion)
                {
                    tCriterionMap[aName] = tCriterion;
                }
            };
            utilities::for_each_sublist(aProblemParams.sublist("Criteria"), tAddCriteria);
        }
        return tCriterionMap;
    }
    else
    {
        return std::map<std::string, CriterionType>{};
    }
}

template <typename ElementType>
void parse_initial_state(Teuchos::ParameterList& aProblemParams,
                         Plato::ScalarVector& aInitialState,
                         Plato::Mesh aMesh,
                         const std::vector<std::string>& aDofNames)
{
    if (aProblemParams.isSublist("Initial State"))
    {
        if (!aProblemParams.isSublist("Computed Fields"))
        {
            ANALYZE_THROWERR("No 'Computed Fields' have been defined");
        }
        const auto tComputedFields = Teuchos::rcp(
            new Plato::ComputedFields<ElementType::mNumSpatialDims>(aMesh, aProblemParams.sublist("Computed Fields")));

        auto tInitStateParams = aProblemParams.sublist("Initial State");
        for (auto i = tInitStateParams.begin(); i != tInitStateParams.end(); ++i)
        {
            const auto& tEntry = tInitStateParams.entry(i);
            const auto& tName = tInitStateParams.name(i);

            if (tEntry.isList())
            {
                auto& tStateList = tInitStateParams.sublist(tName);
                auto tFieldName = tStateList.get<std::string>("Computed Field");
                int tDofIndex = -1;
                for (int j = 0; j < aDofNames.size(); ++j)
                {
                    if (Plato::tolower(aDofNames[j]) == Plato::tolower(tName))
                    {
                        tDofIndex = j;
                    }
                }
                if (tDofIndex == -1)
                {
                    std::stringstream ss;
                    ss << "Tried to initialize non-existent state field: " << Plato::tolower(tName) << std::endl;
                    ss << "Available states are: " << std::endl;
                    for (const auto& tDofName : aDofNames)
                    {
                        ss << "  " << Plato::tolower(tDofName) << std::endl;
                    }
                    ANALYZE_THROWERR(ss.str());
                }
                tComputedFields->get(tFieldName, tDofIndex, aDofNames.size(), aInitialState);
            }
        }
    }
}
}  // namespace plato::parabolic
#endif
