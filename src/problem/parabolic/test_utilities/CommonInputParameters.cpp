#include "problem/parabolic/test_utilities/CommonInputParameters.hpp"

namespace plato::parabolic::test_utilities
{
namespace
{
constexpr auto kMaterialModelName = std::string_view{"tapioca"};
}

Teuchos::ParameterList create_base_thermal_problem_parameters()
{
    Teuchos::ParameterList tParameterList;
    tParameterList.setName("Plato Problem");
    tParameterList.set("PDE Constraint", "Parabolic");
    tParameterList.set("Physics", "Thermal");
    tParameterList.set("Output File", "test_solution_output.txt");

    tParameterList.sublist("Parabolic").sublist("Penalty Function").set("Exponent", 1.0);

    tParameterList.sublist("Spatial Model").sublist("Domains").sublist("Body").set("Element Block", "body");
    tParameterList.sublist("Spatial Model")
        .sublist("Domains")
        .sublist("Body")
        .set("Material Model", std::string{kMaterialModelName});

    tParameterList.sublist("Material Models")
        .sublist(std::string{kMaterialModelName})
        .sublist("Thermal Conduction")
        .set("Thermal Conductivity", 1.0);
    tParameterList.sublist("Material Models")
        .sublist(std::string{kMaterialModelName})
        .sublist("Thermal Mass")
        .set("Temperature Dependent", false);
    tParameterList.sublist("Material Models")
        .sublist(std::string{kMaterialModelName})
        .sublist("Thermal Mass")
        .set("Specific Heat", 1.0);
    tParameterList.sublist("Material Models")
        .sublist(std::string{kMaterialModelName})
        .sublist("Thermal Mass")
        .set("Mass Density", 1.0);

    return tParameterList;
}

void append_time_integrated_state_average_criterion_to_parameter_list(Teuchos::ParameterList& aParamList,
                                                                      const std::string& aName,
                                                                      const std::string& aNodeSet)
{
    aParamList.sublist("Criteria").sublist(aName).set("Type", "Time Integrated State Average");
    aParamList.sublist("Criteria").sublist(aName).set("Nodeset", aNodeSet);
    aParamList.sublist("Criteria").sublist(aName).set("State Component", 0);
}

}  // namespace plato::parabolic::test_utilities
