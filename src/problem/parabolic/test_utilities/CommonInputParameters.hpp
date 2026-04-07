#include <Teuchos_ParameterList.hpp>

namespace plato::parabolic::test_utilities
{
Teuchos::ParameterList create_base_thermal_problem_parameters();

void append_time_integrated_state_average_criterion_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                                      const std::string& aName,
                                                                      const std::string& aNodeSet);
}  // namespace plato::parabolic::test_utilities
