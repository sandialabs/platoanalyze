#include <string_view>

#include "problem/parabolic/TemperatureIntegral_decl.hpp"

#ifdef PLATOANALYZE_USE_EXPLICIT_INSTANTIATION

#include "element/ThermalElement.hpp"
#include "problem/parabolic/ExpInstMacros.hpp"
#include "problem/parabolic/TemperatureIntegral_def.hpp"

PLATO_PARABOLIC_EXP_INST(Plato::Parabolic::TemperatureIntegral, Plato::ThermalElement)

namespace Plato::Parabolic
{
namespace
{
constexpr auto kFunctionName = std::string_view{"temperature integral"};
}
auto temperature_integral_function_name() -> std::string_view { return kFunctionName; }
}  // namespace Plato::Parabolic

#endif
