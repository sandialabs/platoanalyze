#ifndef PLATO_GEOMETRIC_PROBLEM_DEF_H
#define PLATO_GEOMETRIC_PROBLEM_DEF_H

#include <Teuchos_ParameterList.hpp>

#include "domain/SpatialModel.hpp"
#include "mesh/PlatoMesh.hpp"
#include "parsing/TeuchosParsingUtilities.hpp"
#include "problem/Geometrical.hpp"
#include "problem/geometric/Problem_decl.hpp"
#include "problem/geometric/ScalarFunctionBaseFactory.hpp"
#include "utilities/AnalyzeMacros.hpp"
#include "utilities/ParallelComm.hpp"

namespace plato::problem::geometric
{
template <typename PhysicsType>
Problem<PhysicsType>::Problem(Plato::Mesh aMesh, Teuchos::ParameterList& aProblemParams, Plato::Comm::Machine aMachine)
    : mSpatialModel(aMesh, plato::domain::parse_domains(aProblemParams, aMesh), mDataMap)
{
    if (aProblemParams.isSublist("Criteria"))
    {
        auto tAddCriteria = [&tCriteriaMap = mCriteriaMap, &tSpatialModel = mSpatialModel, &tDataMap = mDataMap,
                             &tProblemParams = aProblemParams,
                             tCriterionBaseFactory =
                                 Plato::Geometric::ScalarFunctionBaseFactory<Plato::Geometrical<TopoElementType>>{}](
                                const std::string& aName)
        {
            const auto tCriterion = tCriterionBaseFactory.create(tSpatialModel, tDataMap, tProblemParams, aName);
            if (tCriterion)
            {
                tCriteriaMap[aName] = tCriterion;
            }
        };
        utilities::for_each_sublist(aProblemParams.sublist("Criteria"), tAddCriteria);
    }
}

template <typename PhysicsType>
void Problem<PhysicsType>::updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution)
{
    for (const auto& [tName, tCriterion] : mCriteriaMap)
    {
        tCriterion->updateProblem(aControl);
    }
}

template <typename PhysicsType>
Plato::Solutions Problem<PhysicsType>::solution(const Plato::ScalarVector& aControl)
{
    return Plato::Solutions{};
}

template <typename PhysicsType>
Plato::Scalar Problem<PhysicsType>::criterionValue(const Plato::ScalarVector& aControl,
                                                   const Plato::Solutions& aSolution,
                                                   const std::string& aName)
{
    if (const auto tCriterionIterator = mCriteriaMap.find(aName); tCriterionIterator != mCriteriaMap.end())
    {
        return tCriterionIterator->second->value(aControl);
    }
    else
    {
        ANALYZE_THROWERR(std::string("CRITERION WITH NAME '") + aName + "' IS NOT DEFINED IN THE PROBLEM.")
    }
}

template <typename PhysicsType>
Plato::ScalarVector Problem<PhysicsType>::criterionGradient(const Plato::ScalarVector& aControl,
                                                            const Plato::Solutions& aSolution,
                                                            const std::string& aName)
{
    if (const auto tCriterionIterator = mCriteriaMap.find(aName); tCriterionIterator != mCriteriaMap.end())
    {
        return tCriterionIterator->second->gradient_z(aControl);
    }
    else
    {
        ANALYZE_THROWERR(std::string("CRITERION WITH NAME '") + aName + "' IS NOT DEFINED IN THE PROBLEM.")
    }
}

template <typename PhysicsType>
Plato::ScalarVector Problem<PhysicsType>::criterionGradientX(const Plato::ScalarVector& aControl,
                                                             const Plato::Solutions& aSolution,
                                                             const std::string& aName)
{
    if (const auto tCriterionIterator = mCriteriaMap.find(aName); tCriterionIterator != mCriteriaMap.end())
    {
        return tCriterionIterator->second->gradient_x(aControl);
    }
    else
    {
        ANALYZE_THROWERR(std::string("CRITERION WITH NAME '") + aName + "' IS NOT DEFINED IN THE PROBLEM.")
    }
}

template <typename PhysicsType>
Plato::Solutions Problem<PhysicsType>::getSolution() const
{
    return Plato::Solutions{};
}

template <typename PhysicsType>
void Problem<PhysicsType>::output(const std::filesystem::path& aFilepath) const
{
}
}  // namespace plato::problem::geometric

#endif
