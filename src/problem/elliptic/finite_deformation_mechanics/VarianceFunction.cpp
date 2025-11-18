#include <Kokkos_Core.hpp>
#include <map>
#include <string>

#include "core_types/PlatoTypes.hpp"
#include "domain/SpatialModel.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "problem/elliptic/finite_deformation_mechanics/VarianceFunction_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION
#include "element/BaseExpInstMacros.hpp"
#include "problem/Electromechanics.hpp"
#include "problem/Mechanics.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "problem/elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"
#include "problem/elliptic/finite_deformation_mechanics/VarianceFunction_def.hpp"

PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::VarianceFunction, Plato::Thermal)
PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::VarianceFunction, Plato::Mechanics)
PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::VarianceFunction, Plato::Electromechanics)
PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::VarianceFunction, Plato::Thermomechanics)
PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::VarianceFunction,
                  plato::elliptic::finite_deformation_mechanics::FiniteDeformationMechanics)

#ifdef PLATO_STABILIZED
#include "problem/elliptic/stabilized/Mechanics.hpp"
#include "problem/elliptic/stabilized/Thermomechanics.hpp"
PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::VarianceFunction, Plato::Stabilized::Mechanics)
PLATO_ELEMENT_DEF(plato::elliptic::finite_deformation_mechanics::VarianceFunction, Plato::Stabilized::Thermomechanics)
#endif

namespace plato::elliptic::finite_deformation_mechanics
{
namespace detail
{
Plato::ScalarVector get_last_time_step_state(const Plato::Solutions& aSolution)
{
    const auto tStates = aSolution.get("State");
    const auto tNumSteps = tStates.extent(0);
    return Kokkos::subview(tStates, tNumSteps - 1, Kokkos::ALL());
}

Plato::Scalar compute_field_variance(const Plato::SpatialModel& aSpatialModel,
                                     const std::map<std::string, Plato::ScalarVectorT<Plato::Scalar>>& aDomainResults,
                                     const Plato::Scalar aMean,
                                     const Plato::OrdinalType aNumTotalCells)
{
    Plato::Scalar tVariance{0.0};
    for (const auto& tDomain : aSpatialModel.Domains)
    {
        const auto tNumCells = tDomain.numCells();
        const auto tResult = aDomainResults.at(tDomain.getDomainName());
        Kokkos::parallel_reduce(
            Kokkos::RangePolicy<>(0, tNumCells),
            KOKKOS_LAMBDA(const Plato::OrdinalType tCellOrdinal, Plato::Scalar& aUpdate) {
                aUpdate += (tResult(tCellOrdinal) - aMean) * (tResult(tCellOrdinal) - aMean);
            },
            tVariance);
    }
    return tVariance / aNumTotalCells;
}
}  // namespace detail
}  // namespace plato::elliptic::finite_deformation_mechanics
#endif
