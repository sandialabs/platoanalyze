#include <string_view>

#include "problem/parabolic/InternalThermalEnergy_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/parabolic/ExpInstMacros.hpp"
#include "problem/parabolic/InternalThermalEnergy_def.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::InternalThermalEnergy, Plato::ThermalElement)

namespace Plato::Parabolic
{
namespace
{
constexpr auto kFunctionName = std::string_view{"internal thermal energy"};
}
auto internal_thermal_energy_function_name() -> std::string_view { return kFunctionName; }
}  // namespace Plato::Parabolic

#endif
