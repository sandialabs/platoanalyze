#include "util/PlatoTestHelpers.hpp"

#include "PlatoTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "SpatialModel.hpp"

#include "Tet4.hpp"
#include "Tri3.hpp"

#ifdef PLATO_HEX_ELEMENTS
#include "Quad4.hpp"
#endif

#include "PlatoProblemFactory.hpp"

#include "ImplicitFunctors.hpp"
#include "WorksetBase.hpp"

#include "GradientMatrix.hpp"
#include "CellVolume.hpp"

#include "hyperbolic/EvaluationTypes.hpp"
#include "hyperbolic/micromorphic/MicromorphicMechanicsElement.hpp"
#include "hyperbolic/micromorphic/MicromorphicMechanics.hpp"

#include "material/CubicStiffnessConstant.hpp"
#include "material/TetragonalSkewStiffnessConstant.hpp"

#include "hyperbolic/micromorphic/ElasticModelFactory.hpp"
#include "hyperbolic/micromorphic/InertiaModelFactory.hpp"

#include "hyperbolic/micromorphic/Kinematics.hpp"
#include "hyperbolic/micromorphic/FullStressDivergence.hpp"
#include "hyperbolic/micromorphic/ProjectStressToNode.hpp"
#include "InterpolateFromNodal.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "ProjectToNode.hpp"

#include "hyperbolic/micromorphic/MicromorphicKineticsFactory.hpp"
#include "hyperbolic/micromorphic/AbstractMicromorphicKinetics.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

namespace RelaxedMicromorphicTest
{

Teuchos::RCP<Teuchos::ParameterList>
setup_elastic_model_parameter_list()
{
    return Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "          <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
}

Teuchos::RCP<Teuchos::ParameterList>
setup_elastic_model_expression_parameter_list()
{
    return Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <ParameterList  name='Ce Stiffness Tensor Expression'>   \n"
      "          <Parameter name='symmetry' type='string' value='cubic' /> \n"
      "          <ParameterList  name='Lambda'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{-120.74, -120.74}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Mu'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{557.11, 557.11}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Alpha'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{8.37, 8.37}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cc Stiffness Tensor Expression'>   \n"
      "          <Parameter name='symmetry' type='string' value='tetragonal skew' /> \n"
      "          <ParameterList  name='Mu'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{1.8e-4, 1.8e-4}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cm Stiffness Tensor Expression'>   \n"
      "          <Parameter name='symmetry' type='string' value='cubic' /> \n"
      "          <ParameterList  name='Lambda'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{180.63, 180.63}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Mu'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{255.71, 255.71}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Alpha'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{181.28, 181.28}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
}

Teuchos::RCP<Teuchos::ParameterList>
setup_inertia_model_parameter_list()
{
    return Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "        <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "        <ParameterList  name='Te Inertia Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='0.6'/> <!-- Eta_bar_1 -->  \n"
      "          <Parameter  name='Lambda' type='double' value='2.0'/>  <!-- Eta_bar_3 --> \n"
      "          <Parameter  name='Alpha' type='double' value='0.2'/>  <!-- Eta_bar_star_1 --> \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Tc Inertia Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_bar_2 --> \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Jm Inertia Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='2300.0'/>  <!-- Eta_1 --> \n"
      "          <Parameter  name='Lambda' type='double' value='-1800.0'/>  <!-- Eta_3 --> \n"
      "          <Parameter  name='Alpha' type='double' value='4500.0'/>  <!-- Eta_star_1 --> \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Jc Inertia Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_2 --> \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
}

Teuchos::RCP<Teuchos::ParameterList>
setup_inertia_model_expression_parameter_list()
{
    return Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "        <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "        <ParameterList  name='Te Inertia Tensor Expression'>   \n"
      "          <Parameter name='symmetry' type='string' value='cubic' /> \n"
      "          <ParameterList  name='Lambda'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{2.0, 2.0}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Mu'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{0.6, 0.6}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Alpha'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{0.2, 0.2}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Tc Inertia Tensor Expression'>   \n"
      "          <Parameter name='symmetry' type='string' value='tetragonal skew' /> \n"
      "          <ParameterList  name='Mu'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{1.0e-4, 1.0e-4}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Jm Inertia Tensor Expression'>   \n"
      "          <Parameter name='symmetry' type='string' value='cubic' /> \n"
      "          <ParameterList  name='Lambda'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{-1800.0, -1800.0}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Mu'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{2300.0, 2300.0}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Alpha'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{4500.0, 4500.0}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Jc Inertia Tensor Expression'>   \n"
      "          <Parameter name='symmetry' type='string' value='tetragonal skew' /> \n"
      "          <ParameterList  name='Mu'>   \n"
      "            <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "            <Parameter name='Constant Values' type='Array(double)' value='{1.0e-4, 1.0e-4}'/> \n"
      "            <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "            <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
}

Teuchos::RCP<Teuchos::ParameterList>
setup_full_model_parameter_list_no_inertia()
{
    return Teuchos::getParametersFromXmlString(
      "<ParameterList name='Problem'>                                    \n"
      "  <Parameter name='Physics' type='string' value='Plato Driver' />  \n"
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "            <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "            <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='0.0'/>   \n"
      "          <ParameterList  name='Te Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='0.0'/> <!-- Eta_bar_1 -->  \n"
      "            <Parameter  name='Lambda' type='double' value='0.0'/>  <!-- Eta_bar_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='0.0'/>  <!-- Eta_bar_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Tc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='0.0'/>  <!-- Eta_bar_2 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jm Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='0.0'/>  <!-- Eta_1 --> \n"
      "            <Parameter  name='Lambda' type='double' value='0.0'/>  <!-- Eta_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='0.0'/>  <!-- Eta_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='0.0'/>  <!-- Eta_2 --> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='A-Form' type='bool' value='true'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.0'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
}

Teuchos::RCP<Teuchos::ParameterList>
setup_full_model_parameter_list()
{
    return Teuchos::getParametersFromXmlString(
      "<ParameterList name='Problem'>                                    \n"
      "  <Parameter name='Physics' type='string' value='Plato Driver' />  \n"
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "            <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "            <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <ParameterList  name='Te Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='0.6'/> <!-- Eta_bar_1 -->  \n"
      "            <Parameter  name='Lambda' type='double' value='2.0'/>  <!-- Eta_bar_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='0.2'/>  <!-- Eta_bar_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Tc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_bar_2 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jm Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='2300.0'/>  <!-- Eta_1 --> \n"
      "            <Parameter  name='Lambda' type='double' value='-1800.0'/>  <!-- Eta_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='4500.0'/>  <!-- Eta_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_2 --> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='A-Form' type='bool' value='true'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.0'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
}

Teuchos::RCP<Teuchos::ParameterList>
setup_full_model_expression_parameter_list()
{
    return Teuchos::getParametersFromXmlString(
      "<ParameterList name='Problem'>                                    \n"
      "  <Parameter name='Physics' type='string' value='Plato Driver' />  \n"
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <ParameterList  name='Ce Stiffness Tensor Expression'>   \n"
      "            <Parameter name='symmetry' type='string' value='cubic' /> \n"
      "            <ParameterList  name='Lambda'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{-120.74, -120.74}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "            <ParameterList  name='Mu'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{557.11, 557.11}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "            <ParameterList  name='Alpha'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{8.37, 8.37}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cc Stiffness Tensor Expression'>   \n"
      "            <Parameter name='symmetry' type='string' value='tetragonal skew' /> \n"
      "            <ParameterList  name='Mu'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{1.8e-4, 1.8e-4}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cm Stiffness Tensor Expression'>   \n"
      "            <Parameter name='symmetry' type='string' value='cubic' /> \n"
      "            <ParameterList  name='Lambda'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{180.63, 180.63}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "            <ParameterList  name='Mu'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{255.71, 255.71}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "            <ParameterList  name='Alpha'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{181.28, 181.28}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <ParameterList  name='Te Inertia Tensor Expression'>   \n"
      "            <Parameter name='symmetry' type='string' value='cubic' /> \n"
      "            <ParameterList  name='Lambda'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{2.0, 2.0}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "            <ParameterList  name='Mu'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{0.6, 0.6}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "            <ParameterList  name='Alpha'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{0.2, 0.2}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Tc Inertia Tensor Expression'>   \n"
      "            <Parameter name='symmetry' type='string' value='tetragonal skew' /> \n"
      "            <ParameterList  name='Mu'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{1.0e-4, 1.0e-4}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jm Inertia Tensor Expression'>   \n"
      "            <Parameter name='symmetry' type='string' value='cubic' /> \n"
      "            <ParameterList  name='Lambda'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{-1800.0, -1800.0}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "            <ParameterList  name='Mu'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{2300.0, 2300.0}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "            <ParameterList  name='Alpha'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{4500.0, 4500.0}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jc Inertia Tensor Expression'>   \n"
      "            <Parameter name='symmetry' type='string' value='tetragonal skew' /> \n"
      "            <ParameterList  name='Mu'>   \n"
      "              <Parameter name='Constant Names' type='Array(string)' value='{v0, v1}'/> \n"
      "              <Parameter name='Constant Values' type='Array(double)' value='{1.0e-4, 1.0e-4}'/> \n"
      "              <Parameter name='Independent Variable Name' type='string' value='Z'/> \n"
      "              <Parameter name='Expression' type='string' value='v0+v1*Z'/> \n"
      "            </ParameterList>                                                  \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='NoPenalty'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='A-Form' type='bool' value='true'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.0'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic1D)
{
    auto tParams = setup_elastic_model_parameter_list();

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    constexpr Plato::Scalar tTolerance = 1e-12;

    auto tStiffnessMatrixCe = tMaterialModel->getRank4VoigtConstant("Ce");
    TEST_FLOATING_EQUALITY(993.48, tStiffnessMatrixCe(0,0), tTolerance);

    auto tStiffnessMatrixCc = tMaterialModel->getRank4SkewConstant("Cc");
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(0,0), tTolerance);

    auto tStiffnessMatrixCm = tMaterialModel->getRank4VoigtConstant("Cm");
    TEST_FLOATING_EQUALITY(692.05, tStiffnessMatrixCm(0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic1D_Ce_Lambda_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "        <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "          <Parameter  name='Lamda' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "        </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 1;
    TEST_THROW(Plato::CubicStiffnessConstant<tSpaceDim> tStiffnessMatrixCe(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic2D)
{
    auto tParams = setup_elastic_model_parameter_list();

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    constexpr Plato::Scalar tTolerance = 1e-12;

    auto tStiffnessMatrixCe = tMaterialModel->getRank4VoigtConstant("Ce");
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(8.37,    tStiffnessMatrixCe(2,2), tTolerance);

    auto tStiffnessMatrixCc = tMaterialModel->getRank4SkewConstant("Cc");
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(0,0), tTolerance);

    auto tStiffnessMatrixCm = tMaterialModel->getRank4VoigtConstant("Cm");
    TEST_FLOATING_EQUALITY(692.05, tStiffnessMatrixCm(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(180.63, tStiffnessMatrixCm(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCm(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(180.63, tStiffnessMatrixCm(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(692.05, tStiffnessMatrixCm(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCm(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCm(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCm(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(181.28, tStiffnessMatrixCm(2,2), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic2D_Error_KeyNotInVoigtConstantsMap)
{
    auto tParams = setup_elastic_model_parameter_list();

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    constexpr Plato::Scalar tTolerance = 1e-12;

    TEST_THROW(tMaterialModel->getRank4VoigtConstant("De"), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic2D_Error_KeyNotInSkewConstantsMap)
{
    auto tParams = setup_elastic_model_parameter_list();

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    constexpr Plato::Scalar tTolerance = 1e-12;

    TEST_THROW(tMaterialModel->getRank4SkewConstant("C_c"), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic2D_Cm_Alpha_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "        <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Alpaca' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 2;
    TEST_THROW(Plato::CubicStiffnessConstant<tSpaceDim> tStiffnessMatrixCm(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic3D)
{
    auto tParams = setup_elastic_model_parameter_list();

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    constexpr Plato::Scalar tTolerance = 1e-12;

    auto tStiffnessMatrixCe = tMaterialModel->getRank4VoigtConstant("Ce");
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(0,5), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(1,5), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(-120.74, tStiffnessMatrixCe(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(993.48,  tStiffnessMatrixCe(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(2,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(8.37,    tStiffnessMatrixCe(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(3,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(8.37,    tStiffnessMatrixCe(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(4,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(8.37,    tStiffnessMatrixCe(5,5), tTolerance);

    auto tStiffnessMatrixCc = tMaterialModel->getRank4SkewConstant("Cc");
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffnessMatrixCc(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(1.8e-4, tStiffnessMatrixCc(2,2), tTolerance);

    auto tStiffnessMatrixCm = tMaterialModel->getRank4VoigtConstant("Cm");
    TEST_FLOATING_EQUALITY(692.05,  tStiffnessMatrixCm(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(0,5), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(692.05,  tStiffnessMatrixCm(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(1,5), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(180.63,  tStiffnessMatrixCm(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(692.05,  tStiffnessMatrixCm(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(2,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(181.28,  tStiffnessMatrixCm(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(3,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(181.28,  tStiffnessMatrixCm(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(4,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCm(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(181.28,  tStiffnessMatrixCm(5,5), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic3D_Cc_Mu_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "        <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "          <Parameter  name='Moo' type='double' value='1.8e-4'/>   \n"
      "        </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 3;
    TEST_THROW(Plato::TetragonalSkewStiffnessConstant<tSpaceDim> tStiffnessMatrixCc(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic_Ce_ListName_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <ParameterList  name='Ce Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "          <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 1;

    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    TEST_THROW(tMaterialModelFactory.create("material_1"), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic_Cc_ListName_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "          <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cc Stiffness'>   \n"
      "          <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 1;

    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    TEST_THROW(tMaterialModelFactory.create("material_1"), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, ElasticCubic_Cm_ListName_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                    \n"
      "  <ParameterList name='Material Models'>                           \n"
      "    <ParameterList name='material_1'>                           \n"
      "      <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "        <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "          <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "          <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList  name='See Emm Stiffness Tensor'>   \n"
      "          <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "          <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "          <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "  </ParameterList>                                                  \n"
      "</ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 1;

    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    TEST_THROW(tMaterialModelFactory.create("material_1"), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, InertiaCubic1D)
{
    auto tParams = setup_inertia_model_parameter_list();

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::Hyperbolic::Micromorphic::InertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    constexpr Plato::Scalar tTolerance = 1e-12;

    auto tRho = tInertiaModel->getScalarConstant("Mass Density");
    TEST_FLOATING_EQUALITY(1451.8, tRho, tTolerance);

    auto tInertiaMatrixTe = tInertiaModel->getRank4VoigtConstant("Te");
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(0,0), tTolerance);

    auto tInertiaMatrixTc = tInertiaModel->getRank4SkewConstant("Tc");
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(0,0), tTolerance);

    auto tInertiaMatrixJm = tInertiaModel->getRank4VoigtConstant("Jm");
    TEST_FLOATING_EQUALITY(2800.0, tInertiaMatrixJm(0,0), tTolerance);

    auto tInertiaMatrixJc = tInertiaModel->getRank4SkewConstant("Jc");
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, InertiaCubic1D_Tc_Mu_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "        <ParameterList  name='Tc Inertia Tensor'>   \n"
      "          <Parameter  name='Mew' type='double' value='1.0e-4'/>  <!-- Eta_bar_2 --> \n"
      "        </ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    TEST_THROW(Plato::TetragonalSkewStiffnessConstant<tSpaceDim> tInertiaMatrixTc(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, InertiaCubic2D)
{
    auto tParams = setup_inertia_model_parameter_list();

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::Hyperbolic::Micromorphic::InertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    constexpr Plato::Scalar tTolerance = 1e-12;

    auto tRho = tInertiaModel->getScalarConstant("Mass Density");
    TEST_FLOATING_EQUALITY(1451.8, tRho, tTolerance);

    auto tInertiaMatrixTe = tInertiaModel->getRank4VoigtConstant("Te");
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.2, tInertiaMatrixTe(2,2), tTolerance);

    auto tInertiaMatrixTc = tInertiaModel->getRank4SkewConstant("Tc");
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(0,0), tTolerance);

    auto tInertiaMatrixJm = tInertiaModel->getRank4VoigtConstant("Jm");
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(4500.0,  tInertiaMatrixJm(2,2), tTolerance);

    auto tInertiaMatrixJc = tInertiaModel->getRank4SkewConstant("Jc");
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, InertiaCubic2D_Te_Alpha_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "        <ParameterList  name='Te Inertia Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='0.6'/> <!-- Eta_bar_1 -->  \n"
      "          <Parameter  name='Lambda' type='double' value='2.0'/>  <!-- Eta_bar_3 --> \n"
      "          <Parameter  name='Elpha' type='double' value='0.2'/>  <!-- Eta_bar_star_1 --> \n"
      "        </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 2;
    TEST_THROW(Plato::CubicStiffnessConstant<tSpaceDim> tInertiaMatrixTe(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, InertiaCubic3D)
{
    auto tParams = setup_inertia_model_parameter_list();

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::Hyperbolic::Micromorphic::InertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    constexpr Plato::Scalar tTolerance = 1e-12;

    auto tRho = tInertiaModel->getScalarConstant("Mass Density");
    TEST_FLOATING_EQUALITY(1451.8, tRho, tTolerance);

    auto tInertiaMatrixTe = tInertiaModel->getRank4VoigtConstant("Te");
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(0,5), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(1,5), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(2.0, tInertiaMatrixTe(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(3.2, tInertiaMatrixTe(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(2,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.2, tInertiaMatrixTe(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(3,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.2, tInertiaMatrixTe(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(4,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.2, tInertiaMatrixTe(5,5), tTolerance);

    auto tInertiaMatrixTc = tInertiaModel->getRank4SkewConstant("Tc");
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixTc(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixTc(2,2), tTolerance);

    auto tInertiaMatrixJm = tInertiaModel->getRank4VoigtConstant("Jm");
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(0,5), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(1,5), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(-1800.0, tInertiaMatrixJm(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(2800.0,  tInertiaMatrixJm(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(2,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(4500.0,  tInertiaMatrixJm(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(3,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(4500.0,  tInertiaMatrixJm(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(4,5), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(4500.0,  tInertiaMatrixJm(5,5), tTolerance);

    auto tInertiaMatrixJc = tInertiaModel->getRank4SkewConstant("Jc");
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tInertiaMatrixJc(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(1.0e-4, tInertiaMatrixJc(2,2), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsTests, InertiaCubic3D_Jm_Lambda_Keyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "        <ParameterList  name='Jm Inertia Tensor'>   \n"
      "          <Parameter  name='Mu' type='double' value='2300.0'/>  <!-- Eta_1 --> \n"
      "          <Parameter  name='Lanta' type='double' value='-1800.0'/>  <!-- Eta_3 --> \n"
      "          <Parameter  name='Alpha' type='double' value='4500.0'/>  <!-- Eta_star_1 --> \n"
      "        </ParameterList>                                                  \n"
    );
    constexpr Plato::OrdinalType tSpaceDim = 3;
    TEST_THROW(Plato::CubicStiffnessConstant<tSpaceDim> tInertiaMatrixJm(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsExpressionTests, ElasticCubic2D)
{
    auto tParams = setup_elastic_model_expression_parameter_list();

    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tri3>;
    using EvalType = typename Plato::Hyperbolic::ResidualTypes<ElementType>;

    constexpr Plato::OrdinalType tSpaceDim = ElementType::mNumSpatialDims;
    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    constexpr unsigned int tNumCells = 1;
    constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;
    std::vector<std::vector<Plato::Scalar>> tKnownControl = {{0.2, 0.8, 0.5}};
    Plato::ScalarMultiVectorT<Plato::Scalar> tControl("density", tNumCells, tNumNodesPerCell);
    Plato::TestHelpers::setControlWS(tKnownControl, tControl);

    auto tStiffnessFieldCe = tMaterialModel->template getRank4Field<EvalType>("Ce");
    auto tStiffnessMatrixCe = (*tStiffnessFieldCe)(tControl);
    auto tStiffnessMatrixCe_host = Kokkos::create_mirror_view(tStiffnessMatrixCe);
    Kokkos::deep_copy(tStiffnessMatrixCe_host, tStiffnessMatrixCe);
    TEST_EQUALITY(tStiffnessMatrixCe.extent(2), 3);
    TEST_EQUALITY(tStiffnessMatrixCe.extent(3), 3);

    const Plato::Scalar tTolerance = 1e-12;
    TEST_FLOATING_EQUALITY(1490.22, tStiffnessMatrixCe_host(0,0,0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-181.11, tStiffnessMatrixCe_host(0,0,0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe_host(0,0,0,2), tTolerance);
    TEST_FLOATING_EQUALITY(-181.11, tStiffnessMatrixCe_host(0,0,1,0), tTolerance);
    TEST_FLOATING_EQUALITY(1490.22, tStiffnessMatrixCe_host(0,0,1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe_host(0,0,1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe_host(0,0,2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffnessMatrixCe_host(0,0,2,1), tTolerance);
    TEST_FLOATING_EQUALITY(12.555,  tStiffnessMatrixCe_host(0,0,2,2), tTolerance);

    auto tStiffnessFieldCc = tMaterialModel->template getRank4Field<EvalType>("Cc");
    auto tStiffnessMatrixCc = (*tStiffnessFieldCc)(tControl);
    auto tStiffnessMatrixCc_host = Kokkos::create_mirror_view(tStiffnessMatrixCc);
    Kokkos::deep_copy(tStiffnessMatrixCc_host, tStiffnessMatrixCc);
    TEST_EQUALITY(tStiffnessMatrixCc.extent(2), 1);
    TEST_EQUALITY(tStiffnessMatrixCc.extent(3), 1);

    TEST_FLOATING_EQUALITY(2.7e-4, tStiffnessMatrixCc_host(0,0,0,0), tTolerance);

    auto tStiffnessFieldCm = tMaterialModel->template getRank4Field<EvalType>("Cm");
    auto tStiffnessMatrixCm = (*tStiffnessFieldCm)(tControl);
    auto tStiffnessMatrixCm_host = Kokkos::create_mirror_view(tStiffnessMatrixCm);
    Kokkos::deep_copy(tStiffnessMatrixCm_host, tStiffnessMatrixCm);
    TEST_EQUALITY(tStiffnessMatrixCm.extent(2), 3);
    TEST_EQUALITY(tStiffnessMatrixCm.extent(3), 3);

    TEST_FLOATING_EQUALITY(1038.075, tStiffnessMatrixCm_host(0,0,0,0), tTolerance);
    TEST_FLOATING_EQUALITY(270.945,  tStiffnessMatrixCm_host(0,0,0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,      tStiffnessMatrixCm_host(0,0,0,2), tTolerance);
    TEST_FLOATING_EQUALITY(270.945,  tStiffnessMatrixCm_host(0,0,1,0), tTolerance);
    TEST_FLOATING_EQUALITY(1038.075, tStiffnessMatrixCm_host(0,0,1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,      tStiffnessMatrixCm_host(0,0,1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,      tStiffnessMatrixCm_host(0,0,2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,      tStiffnessMatrixCm_host(0,0,2,1), tTolerance);
    TEST_FLOATING_EQUALITY(271.92,   tStiffnessMatrixCm_host(0,0,2,2), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicMaterialsExpressionTests, InertiaCubic2D)
{
    auto tParams = setup_inertia_model_expression_parameter_list();

    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tri3>;
    using EvalType = typename Plato::Hyperbolic::ResidualTypes<ElementType>;

    constexpr Plato::OrdinalType tSpaceDim = ElementType::mNumSpatialDims;
    Plato::Hyperbolic::Micromorphic::InertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    constexpr unsigned int tNumCells = 1;
    constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;
    std::vector<std::vector<Plato::Scalar>> tKnownControl = {{0.2, 0.8, 0.5}};
    Plato::ScalarMultiVectorT<Plato::Scalar> tControl("density", tNumCells, tNumNodesPerCell);
    Plato::TestHelpers::setControlWS(tKnownControl, tControl);

    const Plato::Scalar tTolerance = 1e-12;
    auto tRho = tInertiaModel->getScalarConstant("Mass Density");
    TEST_FLOATING_EQUALITY(1451.8, tRho, tTolerance);

    auto tInertiaFieldTe = tInertiaModel->template getRank4Field<EvalType>("Te");
    auto tInertiaMatrixTe = (*tInertiaFieldTe)(tControl);
    auto tInertiaMatrixTe_host = Kokkos::create_mirror_view(tInertiaMatrixTe);
    Kokkos::deep_copy(tInertiaMatrixTe_host, tInertiaMatrixTe);
    TEST_EQUALITY(tInertiaMatrixTe.extent(2), 3);
    TEST_EQUALITY(tInertiaMatrixTe.extent(3), 3);

    TEST_FLOATING_EQUALITY(4.8, tInertiaMatrixTe_host(0,0,0,0), tTolerance);
    TEST_FLOATING_EQUALITY(3.0, tInertiaMatrixTe_host(0,0,0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe_host(0,0,0,2), tTolerance);
    TEST_FLOATING_EQUALITY(3.0, tInertiaMatrixTe_host(0,0,1,0), tTolerance);
    TEST_FLOATING_EQUALITY(4.8, tInertiaMatrixTe_host(0,0,1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe_host(0,0,1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe_host(0,0,2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tInertiaMatrixTe_host(0,0,2,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.3, tInertiaMatrixTe_host(0,0,2,2), tTolerance);

    auto tInertiaFieldTc = tInertiaModel->template getRank4Field<EvalType>("Tc");
    auto tInertiaMatrixTc = (*tInertiaFieldTc)(tControl);
    auto tInertiaMatrixTc_host = Kokkos::create_mirror_view(tInertiaMatrixTc);
    Kokkos::deep_copy(tInertiaMatrixTc_host, tInertiaMatrixTc);
    TEST_EQUALITY(tInertiaMatrixTc.extent(2), 1);
    TEST_EQUALITY(tInertiaMatrixTc.extent(3), 1);

    TEST_FLOATING_EQUALITY(1.5e-4, tInertiaMatrixTc_host(0,0,0,0), tTolerance);

    auto tInertiaFieldJm = tInertiaModel->template getRank4Field<EvalType>("Jm");
    auto tInertiaMatrixJm = (*tInertiaFieldJm)(tControl);
    auto tInertiaMatrixJm_host = Kokkos::create_mirror_view(tInertiaMatrixJm);
    Kokkos::deep_copy(tInertiaMatrixJm_host, tInertiaMatrixJm);
    TEST_EQUALITY(tInertiaMatrixJm.extent(2), 3);
    TEST_EQUALITY(tInertiaMatrixJm.extent(3), 3);

    TEST_FLOATING_EQUALITY(4200.0,  tInertiaMatrixJm_host(0,0,0,0), tTolerance);
    TEST_FLOATING_EQUALITY(-2700.0, tInertiaMatrixJm_host(0,0,0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm_host(0,0,0,2), tTolerance);
    TEST_FLOATING_EQUALITY(-2700.0, tInertiaMatrixJm_host(0,0,1,0), tTolerance);
    TEST_FLOATING_EQUALITY(4200.0,  tInertiaMatrixJm_host(0,0,1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm_host(0,0,1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm_host(0,0,2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tInertiaMatrixJm_host(0,0,2,1), tTolerance);
    TEST_FLOATING_EQUALITY(6750.0,  tInertiaMatrixJm_host(0,0,2,2), tTolerance);

    auto tInertiaFieldJc = tInertiaModel->template getRank4Field<EvalType>("Jc");
    auto tInertiaMatrixJc = (*tInertiaFieldJc)(tControl);
    auto tInertiaMatrixJc_host = Kokkos::create_mirror_view(tInertiaMatrixJc);
    Kokkos::deep_copy(tInertiaMatrixJc_host, tInertiaMatrixJc);
    TEST_EQUALITY(tInertiaMatrixJc.extent(2), 1);
    TEST_EQUALITY(tInertiaMatrixJc.extent(3), 1);

    TEST_FLOATING_EQUALITY(1.5e-4, tInertiaMatrixJc_host(0,0,0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, ComputeKinematics)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;

    const int tNumCells = tMesh->NumElements();
    const int tNumNodes = tMesh->NumNodes();
    const auto tCubPoints = ElementType::getCubPoints();
    const auto tNumPoints = ElementType::getCubWeights().size();
    constexpr int tNumDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr int tNumDofsPerNode = ElementType::mNumDofsPerNode;
    constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;
    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;

    int tNumDofs = tNumNodes*tNumDofsPerNode;
    Plato::ScalarVector tState("state", tNumDofs);
    Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
    {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tState(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-7)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
    });
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tNumDofsPerCell);
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    tWorksetBase.worksetState(tState, tStateWS);

    std::vector<Plato::Scalar> tKnownGradients = { 
      0.0, -2.0, 0.0,   2.0, 0.0, -2.0,   -2.0, 2.0, 0.0,   0.0, 0.0, 2.0,
      0.0, -2.0, 0.0,   0.0, 2.0, -2.0,   -2.0, 0.0, 2.0,   2.0, 0.0, 0.0,
      0.0, 0.0, -2.0,   -2.0, 2.0, 0.0,   0.0, -2.0, 2.0,   2.0, 0.0, 0.0,
      0.0, 0.0, -2.0,   -2.0, 0.0, 2.0,   2.0, -2.0, 0.0,   0.0, 2.0, 0.0,
      -2.0, 0.0, 0.0,   0.0, -2.0, 2.0,   2.0, 0.0, -2.0,   0.0, 2.0, 0.0,
      -2.0, 0.0, 0.0,   2.0, -2.0, 0.0,   0.0, 2.0, -2.0,   0.0, 0.0, 2.0
    }; // tNumCells x tNumNodesPerCell x tSpaceDim
    auto tGradientVals = Plato::TestHelpers::create_device_view(tKnownGradients);
    Plato::ScalarArray4DT<Plato::Scalar> tGradients("",tNumCells,tNumPoints,tNumNodesPerCell,tSpaceDim);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iNode=0; iNode<tNumNodesPerCell; iNode++)
          for(int iDim=0; iDim<tSpaceDim; iDim++)
            tGradients(iCell,iPoint,iNode,iDim) = tGradientVals(iCell*tNumNodesPerCell*tSpaceDim+iNode*tSpaceDim+iDim);
    });

    Plato::ScalarArray3DT<Plato::Scalar> tSymDisplacementGradients ("strain",tNumCells,tNumPoints,tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwDisplacementGradients ("strain",tNumCells,tNumPoints,tNumSkwTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSymMicroDistortionTensors("strain",tNumCells,tNumPoints,tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwMicroDistortionTensors("strain",tNumCells,tNumPoints,tNumSkwTerms);

    Plato::Hyperbolic::Micromorphic::Kinematics<ElementType> computeKinematics;

    Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
    {
        auto tCubPoint = tCubPoints(gpOrdinal);
        auto tBasisValues = ElementType::basisValues(tCubPoint);

        computeKinematics(cellOrdinal,gpOrdinal,tSymDisplacementGradients,tSkwDisplacementGradients,tSymMicroDistortionTensors,tSkwMicroDistortionTensors,tStateWS,tBasisValues,tGradients);
    });

    constexpr unsigned int iPoint = 0;
    // test symmetric displacement gradient
    //
    auto tSymDisplacementGradient_Host = Kokkos::create_mirror_view( tSymDisplacementGradients );
    Kokkos::deep_copy( tSymDisplacementGradient_Host, tSymDisplacementGradients );

    std::vector<std::vector<Plato::Scalar>> tSymDisplacementGradient_Gold = { 
      {8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06}, 
      {8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06}, 
      {8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06}, 
      {8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06}, 
      {8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06}, 
      {8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06}, 
    };

    for(int iCell=0; iCell<int(tSymDisplacementGradient_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymDisplacementGradient_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymDisplacementGradient_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymDisplacementGradient_Host(iCell,iPoint,iVoigt), tSymDisplacementGradient_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }
  
    // test skew displacement gradient
    //
    auto tSkwDisplacementGradient_Host = Kokkos::create_mirror_view( tSkwDisplacementGradients );
    Kokkos::deep_copy( tSkwDisplacementGradient_Host, tSkwDisplacementGradients );

    std::vector<std::vector<Plato::Scalar>> tSkwDisplacementGradient_Gold = { 
      {-8e-07, -2.2e-06, -1.2e-06}, 
      {-8e-07, -2.2e-06, -1.2e-06}, 
      {-8e-07, -2.2e-06, -1.2e-06}, 
      {-8e-07, -2.2e-06, -1.2e-06}, 
      {-8e-07, -2.2e-06, -1.2e-06}, 
      {-8e-07, -2.2e-06, -1.2e-06}, 
    };

    for(int iCell=0; iCell<int(tSkwDisplacementGradient_Gold.size()); iCell++){
      for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++){
        if(tSkwDisplacementGradient_Gold[iCell][iSkw] == 0.0){
          TEST_ASSERT(fabs(tSkwDisplacementGradient_Host(iCell,iPoint,iSkw)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSkwDisplacementGradient_Host(iCell,iPoint,iSkw), tSkwDisplacementGradient_Gold[iCell][iSkw], 1e-13);
        }
      }
    }
  
    // test symmetric micro distortion tensor
    //
    auto tSymMicroDistortionTensor_Host = Kokkos::create_mirror_view( tSymMicroDistortionTensors );
    Kokkos::deep_copy( tSymMicroDistortionTensor_Host, tSymMicroDistortionTensors );
       
    std::vector<std::vector<Plato::Scalar>>
      tSymMicroDistortionTensor_Gold = { 
        {1.90249223594997e-06, 2.37811529493746e-06, 2.85373835392495e-06, 8.08559200278735e-06, 9.03683812076233e-06, 9.98808423873732e-06},
        {1.02111456180002e-06, 1.27639320225002e-06, 1.53167184270003e-06, 4.33973688765008e-06, 4.85029416855009e-06, 5.36085144945010e-06},
        {1.14472135955000e-06, 1.43090169943750e-06, 1.71708203932500e-06, 4.86506577808749e-06, 5.43742645786249e-06, 6.00978713763749e-06},
        {8.97507764050041e-07, 1.12188470506255e-06, 1.34626164607506e-06, 3.81440799721267e-06, 4.26316187923769e-06, 4.71191576126271e-06},
        {1.77888543819999e-06, 2.22360679774998e-06, 2.66832815729998e-06, 7.56026311234994e-06, 8.44970583144994e-06, 9.33914855054993e-06},
        {1.65527864045001e-06, 2.06909830056251e-06, 2.48291796067501e-06, 7.03493422191253e-06, 7.86257354213754e-06, 8.69021286236254e-06}
      };

    for(int iCell=0; iCell<int(tSymMicroDistortionTensor_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymMicroDistortionTensor_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymMicroDistortionTensor_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymMicroDistortionTensor_Host(iCell,iPoint,iVoigt), tSymMicroDistortionTensor_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }
  
    // test skew micro distortion tensor
    //
    auto tSkwMicroDistortionTensor_Host = Kokkos::create_mirror_view( tSkwMicroDistortionTensors );
    Kokkos::deep_copy( tSkwMicroDistortionTensor_Host, tSkwMicroDistortionTensors );

    std::vector<std::vector<Plato::Scalar>>
      tSkwMicroDistortionTensor_Gold = { 
        {-1.42686917696247e-06, -1.42686917696247e-06, -1.42686917696247e-06},
        {-7.65835921350014e-07, -7.65835921350015e-07, -7.65835921350015e-07},
        {-8.58541019662499e-07, -8.58541019662499e-07, -8.58541019662499e-07},
        {-6.73130823037530e-07, -6.73130823037530e-07, -6.73130823037530e-07},
        {-1.33416407864999e-06, -1.33416407864999e-06, -1.33416407864999e-06},
        {-1.24145898033751e-06, -1.24145898033751e-06, -1.24145898033751e-06}
      };

    for(int iCell=0; iCell<int(tSkwMicroDistortionTensor_Gold.size()); iCell++){
      for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++){
        if(tSkwMicroDistortionTensor_Gold[iCell][iSkw] == 0.0){
          TEST_ASSERT(fabs(tSkwMicroDistortionTensor_Host(iCell,iPoint,iSkw)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSkwMicroDistortionTensor_Host(iCell,iPoint,iSkw), tSkwMicroDistortionTensor_Gold[iCell][iSkw], 1e-13);
        }
      }
    }
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, ComputeLinearElasticKinetics)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    using EvalType = typename Plato::Hyperbolic::Evaluation<ElementType>::Residual;

    const int tNumCells = tMesh->NumElements();
    const auto tNumPoints = ElementType::getCubWeights().size();
    TEST_EQUALITY(tNumCells, 6);
    TEST_EQUALITY(tNumPoints, 4);

    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;

    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("Control Workset", tNumCells, ElementType::mNumNodesPerCell);

    std::vector<Plato::Scalar> tKnownSymDisplacementGradients = { 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
    }; // tNumCells x tNumVoigtTerms
    auto tSymDisplacementGradientVals = Plato::TestHelpers::create_device_view(tKnownSymDisplacementGradients);
    Plato::ScalarArray3DT<Plato::Scalar> tSymDisplacementGradients("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymDisplacementGradients(iCell,iPoint,iVoigt) = tSymDisplacementGradientVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwDisplacementGradients = { 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
    }; // tNumCells x tNumSkwTerms
    auto tSkwDisplacementGradientVals = Plato::TestHelpers::create_device_view(tKnownSkwDisplacementGradients);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwDisplacementGradients("", tNumCells, tNumPoints, tNumSkwTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++)
          tSkwDisplacementGradients(iCell,iPoint,iSkw) = tSkwDisplacementGradientVals(iCell*tNumSkwTerms+iSkw);
    });

    std::vector<Plato::Scalar>
      tKnownSymMicroDistortionTensors = { 
        1.90249223594997e-06, 2.37811529493746e-06, 2.85373835392495e-06, 8.08559200278735e-06, 9.03683812076233e-06, 9.98808423873732e-06,
        1.02111456180002e-06, 1.27639320225002e-06, 1.53167184270003e-06, 4.33973688765008e-06, 4.85029416855009e-06, 5.36085144945010e-06,
        1.14472135955000e-06, 1.43090169943750e-06, 1.71708203932500e-06, 4.86506577808749e-06, 5.43742645786249e-06, 6.00978713763749e-06,
        8.97507764050041e-07, 1.12188470506255e-06, 1.34626164607506e-06, 3.81440799721267e-06, 4.26316187923769e-06, 4.71191576126271e-06,
        1.77888543819999e-06, 2.22360679774998e-06, 2.66832815729998e-06, 7.56026311234994e-06, 8.44970583144994e-06, 9.33914855054993e-06,
        1.65527864045001e-06, 2.06909830056251e-06, 2.48291796067501e-06, 7.03493422191253e-06, 7.86257354213754e-06, 8.69021286236254e-06
      };

    auto tSymMicroDistortionTensorVals = Plato::TestHelpers::create_device_view(tKnownSymMicroDistortionTensors);
    Plato::ScalarArray3DT<Plato::Scalar> tSymMicroDistortionTensors("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymMicroDistortionTensors(iCell,iPoint,iVoigt) = tSymMicroDistortionTensorVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar>
      tKnownSkwMicroDistortionTensors = { 
        -1.42686917696247e-06, -1.42686917696247e-06, -1.42686917696247e-06,
        -7.65835921350014e-07, -7.65835921350015e-07, -7.65835921350015e-07,
        -8.58541019662499e-07, -8.58541019662499e-07, -8.58541019662499e-07,
        -6.73130823037530e-07, -6.73130823037530e-07, -6.73130823037530e-07,
        -1.33416407864999e-06, -1.33416407864999e-06, -1.33416407864999e-06,
        -1.24145898033751e-06, -1.24145898033751e-06, -1.24145898033751e-06
      };

    auto tSkwMicroDistortionTensorVals = Plato::TestHelpers::create_device_view(tKnownSkwMicroDistortionTensors);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwMicroDistortionTensors("", tNumCells, tNumPoints, tNumSkwTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++)
          tSkwMicroDistortionTensors(iCell,iPoint,iSkw) = tSkwMicroDistortionTensorVals(iCell*tNumSkwTerms+iSkw);
    });

    auto tParams = setup_elastic_model_parameter_list();
    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    Plato::Hyperbolic::Micromorphic::MicromorphicKineticsFactory<EvalType, ElementType> tKineticsFactory;
    auto tKinetics = tKineticsFactory.create(tMaterialModel);

    Plato::ScalarArray3DT<Plato::Scalar> tSymCauchyStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwCauchyStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSymMicroStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);

    // Compute kinetics
    (*tKinetics)(tSymCauchyStresses, tSkwCauchyStresses, tSymMicroStresses, tSymDisplacementGradients, tSkwDisplacementGradients, tSymMicroDistortionTensors, tSkwMicroDistortionTensors, tControlWS);

    constexpr unsigned int iPoint = 0;
    // test symmetric cauchy stress
    //
    auto tSymCauchyStress_Host = Kokkos::create_mirror_view( tSymCauchyStresses );
    Kokkos::deep_copy( tSymCauchyStress_Host, tSymCauchyStresses );

    std::vector<std::vector<Plato::Scalar>>
      tSymCauchyStress_Gold = { 
        {-0.000632645977007925, -0.00116259470179297,  -0.00191538742657801,  -5.42844050633301e-05, -5.38763350707807e-05, -6.68602650782313e-05},
        {-4.96631213298136e-05, -0.000334099688092018, -0.000841380254854222, -2.29315977496312e-05, -1.88349621907643e-05, -2.81303266318973e-05},
        {-0.000131422219667548, -0.000450290077976998, -0.000992001936286448, -2.73286005625923e-05, -2.37492594523090e-05, -3.35619183420258e-05},
        { 3.20959770079209e-05, -0.000217909298207038, -0.000690758573421997, -1.85345949366701e-05, -1.39206649292195e-05, -2.26987349217689e-05},
        {-0.000550886878670190, -0.00104640431190799,  -0.00176476574514578,  -4.98874022503690e-05, -4.89620378092360e-05, -6.14286733681029e-05},
        {-0.000469127780332455, -0.000930213922023007, -0.00161414406371356,  -4.54903994374079e-05, -4.40477405476912e-05, -5.59970816579745e-05}
      };

    for(int iCell=0; iCell<int(tSymCauchyStress_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymCauchyStress_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymCauchyStress_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymCauchyStress_Host(iCell,iPoint,iVoigt), tSymCauchyStress_Gold[iCell][iVoigt], 1e-12);
        }
      }
    }

    // test skew cauchy stress
    //
    auto tSkwCauchyStress_Host = Kokkos::create_mirror_view( tSkwCauchyStresses );
    Kokkos::deep_copy( tSkwCauchyStress_Host, tSkwCauchyStresses );

    std::vector<std::vector<Plato::Scalar>>
      tSkwCauchyStress_Gold = { 
        {0, 0, 0,  1.12836451853245e-10, -1.39163548146755e-10,  4.08364518532452e-11},
        {0, 0, 0, -6.14953415699743e-12, -2.58149534156997e-10, -7.81495341569974e-11},
        {0, 0, 0,  1.05373835392498e-11, -2.41462616460750e-10, -6.14626164607503e-11},
        {0, 0, 0, -2.28364518532444e-11, -2.74836451853245e-10, -9.48364518532445e-11},
        {0, 0, 0,  9.61495341569981e-11, -1.55850465843002e-10,  2.41495341569981e-11},
        {0, 0, 0,  7.94626164607511e-11, -1.72537383539249e-10,  7.46261646075093e-12}
      };

    for(int iCell=0; iCell<int(tSkwCauchyStress_Gold.size()); iCell++){
      for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
        if(tSkwCauchyStress_Gold[iCell][iSkw] == 0.0){
          TEST_ASSERT(fabs(tSkwCauchyStress_Host(iCell,iPoint,iSkw)) < 1e-14);
        } else {
          TEST_FLOATING_EQUALITY(tSkwCauchyStress_Host(iCell,iPoint,iSkw), tSkwCauchyStress_Gold[iCell][iSkw], 1e-12);
        }
      }
    }
  
    // test symmetric micro stress
    //
    auto tSymMicroStress_Host = Kokkos::create_mirror_view( tSymMicroStresses );
    Kokkos::deep_copy( tSymMicroStress_Host, tSymMicroStresses );

    std::vector<std::vector<Plato::Scalar>>
      tSymMicroStress_Gold = { 
        {0.00226164947648319, 0.00250489262131057, 0.00274813576613796, 0.00146575611826529,  0.00163819801453180,  0.00181063991079830},
        {0.00121388312156303, 0.00134443772386197, 0.00147499232616091, 0.000786707502993207, 0.000879261326874760, 0.000971815150756314},
        {0.00136082471960925, 0.00150718306903451, 0.00165354141845978, 0.000881939124251701, 0.000985696668281312, 0.00108945421231092},
        {0.00106694152351682, 0.00118169237868944, 0.00129644323386205, 0.000691475881734713, 0.000772825985468209, 0.000854176089201704},
        {0.00211470787843698, 0.00234214727613803, 0.00256958667383909, 0.00137052449700680,  0.00153176267312524,  0.00169300084924369},
        {0.00196776628039076, 0.00217940193096550, 0.00239103758154023, 0.00127529287574830,  0.00142532733171869,  0.00157536178768908}
      };
    

    for(int iCell=0; iCell<int(tSymMicroStress_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymMicroStress_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymMicroStress_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymMicroStress_Host(iCell,iPoint,iVoigt), tSymMicroStress_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, ComputeExpressionElasticKinetics)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    using EvalType = typename Plato::Hyperbolic::Evaluation<ElementType>::Residual;

    const int tNumCells = tMesh->NumElements();
    const auto tNumPoints = ElementType::getCubWeights().size();

    if (tNumPoints != 1 && tNumPoints != 4) {
      throw std::logic_error("This test only works for 1-point or 4-point quadrature.");
    }

    TEST_EQUALITY(tNumCells, 6);
    TEST_EQUALITY(tNumPoints, 4);

    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;

    std::vector<std::vector<Plato::Scalar>> tKnownControl = {
      {0.0, 1.0, 0.5, 0.5},
      {0.8, 1.0, 0.5, 0.7},
      {0.4, 0.05, 0.3, 0.25}};
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("density", tNumCells, ElementType::mNumNodesPerCell);
    Plato::TestHelpers::setControlWS(tKnownControl, tControlWS);

    std::vector<Plato::Scalar>
      tControlWeights = {0.723606797749980, 0.861803398874990, 0.160557280900008, 0.0, 0.0, 0.0};

    std::vector<Plato::Scalar> tKnownSymDisplacementGradients = { 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
      8e-07, 8e-07, 6e-07, 1.6e-06, 2.6e-06, 2e-06, 
    }; // tNumCells x tNumVoigtTerms
    auto tSymDisplacementGradientVals = Plato::TestHelpers::create_device_view(tKnownSymDisplacementGradients);
    Plato::ScalarArray3DT<Plato::Scalar> tSymDisplacementGradients("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymDisplacementGradients(iCell,iPoint,iVoigt) = tSymDisplacementGradientVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwDisplacementGradients = { 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
      -8e-07, -2.2e-06, -1.2e-06, 
    }; // tNumCells x tNumSkwTerms
    auto tSkwDisplacementGradientVals = Plato::TestHelpers::create_device_view(tKnownSkwDisplacementGradients);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwDisplacementGradients("", tNumCells, tNumPoints, tNumSkwTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++)
          tSkwDisplacementGradients(iCell,iPoint,iSkw) = tSkwDisplacementGradientVals(iCell*tNumSkwTerms+iSkw);
    });

    std::vector<Plato::Scalar>
      tKnownSymMicroDistortionTensors = { 
        1.90249223594997e-06, 2.37811529493746e-06, 2.85373835392495e-06, 8.08559200278735e-06, 9.03683812076233e-06, 9.98808423873732e-06,
        1.02111456180002e-06, 1.27639320225002e-06, 1.53167184270003e-06, 4.33973688765008e-06, 4.85029416855009e-06, 5.36085144945010e-06,
        1.14472135955000e-06, 1.43090169943750e-06, 1.71708203932500e-06, 4.86506577808749e-06, 5.43742645786249e-06, 6.00978713763749e-06,
        8.97507764050041e-07, 1.12188470506255e-06, 1.34626164607506e-06, 3.81440799721267e-06, 4.26316187923769e-06, 4.71191576126271e-06,
        1.77888543819999e-06, 2.22360679774998e-06, 2.66832815729998e-06, 7.56026311234994e-06, 8.44970583144994e-06, 9.33914855054993e-06,
        1.65527864045001e-06, 2.06909830056251e-06, 2.48291796067501e-06, 7.03493422191253e-06, 7.86257354213754e-06, 8.69021286236254e-06
      };

    auto tSymMicroDistortionTensorVals = Plato::TestHelpers::create_device_view(tKnownSymMicroDistortionTensors);
    Plato::ScalarArray3DT<Plato::Scalar> tSymMicroDistortionTensors("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymMicroDistortionTensors(iCell,iPoint,iVoigt) = tSymMicroDistortionTensorVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar>
      tKnownSkwMicroDistortionTensors = { 
        -1.42686917696247e-06, -1.42686917696247e-06, -1.42686917696247e-06,
        -7.65835921350014e-07, -7.65835921350015e-07, -7.65835921350015e-07,
        -8.58541019662499e-07, -8.58541019662499e-07, -8.58541019662499e-07,
        -6.73130823037530e-07, -6.73130823037530e-07, -6.73130823037530e-07,
        -1.33416407864999e-06, -1.33416407864999e-06, -1.33416407864999e-06,
        -1.24145898033751e-06, -1.24145898033751e-06, -1.24145898033751e-06
      };

    auto tSkwMicroDistortionTensorVals = Plato::TestHelpers::create_device_view(tKnownSkwMicroDistortionTensors);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwMicroDistortionTensors("", tNumCells, tNumPoints, tNumSkwTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++)
          tSkwMicroDistortionTensors(iCell,iPoint,iSkw) = tSkwMicroDistortionTensorVals(iCell*tNumSkwTerms+iSkw);
    });

    auto tParams = setup_elastic_model_expression_parameter_list();
    Plato::Hyperbolic::Micromorphic::ElasticModelFactory<tSpaceDim> tMaterialModelFactory(*tParams);
    auto tMaterialModel = tMaterialModelFactory.create("material_1");

    Plato::Hyperbolic::Micromorphic::MicromorphicKineticsFactory<EvalType, ElementType> tKineticsFactory;
    auto tKinetics = tKineticsFactory.create(tMaterialModel);

    Plato::ScalarArray3DT<Plato::Scalar> tSymCauchyStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwCauchyStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSymMicroStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);

    // Compute kinetics
    (*tKinetics)(tSymCauchyStresses, tSkwCauchyStresses, tSymMicroStresses, tSymDisplacementGradients, tSkwDisplacementGradients, tSymMicroDistortionTensors, tSkwMicroDistortionTensors, tControlWS);

    constexpr unsigned int iPoint = 0;
    // test symmetric cauchy stress
    //
    auto tSymCauchyStress_Host = Kokkos::create_mirror_view( tSymCauchyStresses );
    Kokkos::deep_copy( tSymCauchyStress_Host, tSymCauchyStresses );

    std::vector<std::vector<Plato::Scalar>>
      tSymCauchyStress_Gold = { 
        {-0.000632645977007925, -0.00116259470179297,  -0.00191538742657801,  -5.42844050633301e-05, -5.38763350707807e-05, -6.68602650782313e-05},
        {-4.96631213298136e-05, -0.000334099688092018, -0.000841380254854222, -2.29315977496312e-05, -1.88349621907643e-05, -2.81303266318973e-05},
        {-0.000131422219667548, -0.000450290077976998, -0.000992001936286448, -2.73286005625923e-05, -2.37492594523090e-05, -3.35619183420258e-05},
        { 3.20959770079209e-05, -0.000217909298207038, -0.000690758573421997, -1.85345949366701e-05, -1.39206649292195e-05, -2.26987349217689e-05},
        {-0.000550886878670190, -0.00104640431190799,  -0.00176476574514578,  -4.98874022503690e-05, -4.89620378092360e-05, -6.14286733681029e-05},
        {-0.000469127780332455, -0.000930213922023007, -0.00161414406371356,  -4.54903994374079e-05, -4.40477405476912e-05, -5.59970816579745e-05}
      };

    // add weighting due to expression: E0*(1+Z)
    for (int i=0; i<tSymCauchyStress_Gold.size(); i++) {
      for (auto& tVal : tSymCauchyStress_Gold[i]) {
        tVal *= (1.0+tControlWeights[i]);
      }
    }

    for(int iCell=0; iCell<int(tSymCauchyStress_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymCauchyStress_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymCauchyStress_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymCauchyStress_Host(iCell,iPoint,iVoigt), tSymCauchyStress_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }

    // test skew cauchy stress
    //
    auto tSkwCauchyStress_Host = Kokkos::create_mirror_view( tSkwCauchyStresses );
    Kokkos::deep_copy( tSkwCauchyStress_Host, tSkwCauchyStresses );

    std::vector<std::vector<Plato::Scalar>>
      tSkwCauchyStress_Gold = { 
        {0, 0, 0,  1.12836451853245e-10, -1.39163548146755e-10,  4.08364518532452e-11},
        {0, 0, 0, -6.14953415699743e-12, -2.58149534156997e-10, -7.81495341569974e-11},
        {0, 0, 0,  1.05373835392498e-11, -2.41462616460750e-10, -6.14626164607503e-11},
        {0, 0, 0, -2.28364518532444e-11, -2.74836451853245e-10, -9.48364518532445e-11},
        {0, 0, 0,  9.61495341569981e-11, -1.55850465843002e-10,  2.41495341569981e-11},
        {0, 0, 0,  7.94626164607511e-11, -1.72537383539249e-10,  7.46261646075093e-12}
      };

    // add weighting due to expression: E0*(1+Z)
    for (int i=0; i<tSkwCauchyStress_Gold.size(); i++) {
      for (auto& tVal : tSkwCauchyStress_Gold[i]) {
        tVal *= (1.0+tControlWeights[i]);
      }
    }

    for(int iCell=0; iCell<int(tSkwCauchyStress_Gold.size()); iCell++){
      for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
        if(tSkwCauchyStress_Gold[iCell][iSkw] == 0.0){
          TEST_ASSERT(fabs(tSkwCauchyStress_Host(iCell,iPoint,iSkw)) < 1e-14);
        } else {
          TEST_FLOATING_EQUALITY(tSkwCauchyStress_Host(iCell,iPoint,iSkw), tSkwCauchyStress_Gold[iCell][iSkw], 1e-12);
        }
      }
    }
  
    // test symmetric micro stress
    //
    auto tSymMicroStress_Host = Kokkos::create_mirror_view( tSymMicroStresses );
    Kokkos::deep_copy( tSymMicroStress_Host, tSymMicroStresses );

    std::vector<std::vector<Plato::Scalar>>
      tSymMicroStress_Gold = { 
        {0.00226164947648319, 0.00250489262131057, 0.00274813576613796, 0.00146575611826529,  0.00163819801453180,  0.00181063991079830},
        {0.00121388312156303, 0.00134443772386197, 0.00147499232616091, 0.000786707502993207, 0.000879261326874760, 0.000971815150756314},
        {0.00136082471960925, 0.00150718306903451, 0.00165354141845978, 0.000881939124251701, 0.000985696668281312, 0.00108945421231092},
        {0.00106694152351682, 0.00118169237868944, 0.00129644323386205, 0.000691475881734713, 0.000772825985468209, 0.000854176089201704},
        {0.00211470787843698, 0.00234214727613803, 0.00256958667383909, 0.00137052449700680,  0.00153176267312524,  0.00169300084924369},
        {0.00196776628039076, 0.00217940193096550, 0.00239103758154023, 0.00127529287574830,  0.00142532733171869,  0.00157536178768908}
      };

    // add weighting due to expression: E0*(1+Z)
    for (int i=0; i<tSymMicroStress_Gold.size(); i++) {
      for (auto& tVal : tSymMicroStress_Gold[i]) {
        tVal *= (1.0+tControlWeights[i]);
      }
    }

    for(int iCell=0; iCell<int(tSymMicroStress_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymMicroStress_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymMicroStress_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymMicroStress_Host(iCell,iPoint,iVoigt), tSymMicroStress_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, ComputeLinearInertiaKinetics)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    using EvalType = typename Plato::Hyperbolic::Evaluation<ElementType>::Residual;

    const int tNumCells = tMesh->NumElements();
    const auto tNumPoints = ElementType::getCubWeights().size();
    TEST_EQUALITY(tNumCells, 6);
    TEST_EQUALITY(tNumPoints, 4);

    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;

    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("Control Workset", tNumCells, ElementType::mNumNodesPerCell);

    std::vector<Plato::Scalar> tKnownSymGradientMicroInertia = { 
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002
    }; // tNumCells x tNumVoigtTerms
    auto tSymGradientMicroInertiaVals = Plato::TestHelpers::create_device_view(tKnownSymGradientMicroInertia);
    Plato::ScalarArray3DT<Plato::Scalar> tSymGradientMicroInertias("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymGradientMicroInertias(iCell,iPoint,iVoigt) = tSymGradientMicroInertiaVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwGradientMicroInertia = { 
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012
    }; // tNumCells x tNumSkwTerms // tNumCells x tNumSkwTerms
    auto tSkwGradientMicroInertiaVals = Plato::TestHelpers::create_device_view(tKnownSkwGradientMicroInertia);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwGradientMicroInertias("", tNumCells, tNumPoints, tNumSkwTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++)
          tSkwGradientMicroInertias(iCell,iPoint,iSkw) = tSkwGradientMicroInertiaVals(iCell*tNumSkwTerms+iSkw);
    });

    std::vector<Plato::Scalar> tKnownSymFreeMicroInertia = { 
      0.00015, 0.0001875, 0.000225, 0.0006375, 0.0007125, 0.0007875,
      0.00012,   0.00015,  0.00018,   0.00051,   0.00057,   0.00063,
      0.00011, 0.0001375, 0.000165, 0.0004675, 0.0005225, 0.0005775,
      0.00013, 0.0001625, 0.000195, 0.0005525, 0.0006175, 0.0006825,
      0.00016,    0.0002,  0.00024,   0.00068,   0.00076,   0.00084,
      0.00017, 0.0002125, 0.000255, 0.0007225, 0.0008075, 0.0008925
    }; // tNumCells x tNumVoigtTerms
    auto tSymFreeMicroInertiaVals = Plato::TestHelpers::create_device_view(tKnownSymFreeMicroInertia);
    Plato::ScalarArray3DT<Plato::Scalar> tSymFreeMicroInertias("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymFreeMicroInertias(iCell,iPoint,iVoigt) = tSymFreeMicroInertiaVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwFreeMicroInertia = { 
      -0.0001125, -0.0001125, -0.0001125,
          -9e-05,     -9e-05,     -9e-05,
       -8.25e-05,  -8.25e-05,  -8.25e-05,
       -9.75e-05,  -9.75e-05,  -9.75e-05,
        -0.00012,   -0.00012,   -0.00012,
      -0.0001275, -0.0001275, -0.0001275
    }; // tNumCells x tNumSkwTerms
    auto tSkwFreeMicroInertiaVals = Plato::TestHelpers::create_device_view(tKnownSkwFreeMicroInertia);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwFreeMicroInertias("", tNumCells, tNumPoints, tNumSkwTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++)
          tSkwFreeMicroInertias(iCell,iPoint,iSkw) = tSkwFreeMicroInertiaVals(iCell*tNumSkwTerms+iSkw);
    });

    auto tParams = setup_inertia_model_parameter_list();
    Plato::Hyperbolic::Micromorphic::InertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    Plato::Hyperbolic::Micromorphic::MicromorphicKineticsFactory<EvalType, ElementType> tKineticsFactory;
    auto tKinetics = tKineticsFactory.create(tInertiaModel);

    Plato::ScalarArray3DT<Plato::Scalar> tSymGradientInertiaStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwGradientInertiaStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSymFreeInertiaStresses    ("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwFreeInertiaStresses    ("stress", tNumCells, tNumPoints, tNumVoigtTerms);

    // Compute kinetics
    (*tKinetics)(tSymGradientInertiaStresses, tSkwGradientInertiaStresses, tSymFreeInertiaStresses, tSkwFreeInertiaStresses, tSymGradientMicroInertias, tSkwGradientMicroInertias, tSymFreeMicroInertias, tSkwFreeMicroInertias, tControlWS);

    constexpr unsigned int iPoint = 0;
    // test symmetric gradient inertia stress
    //
    auto tSymGradientInertiaStress_Host = Kokkos::create_mirror_view( tSymGradientInertiaStresses );
    Kokkos::deep_copy( tSymGradientInertiaStress_Host, tSymGradientInertiaStresses );

    std::vector<std::vector<Plato::Scalar>> tSymGradientInertiaStress_Gold = { 
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05}
    };

    for(int iCell=0; iCell<int(tSymGradientInertiaStress_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymGradientInertiaStress_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymGradientInertiaStress_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymGradientInertiaStress_Host(iCell,iPoint,iVoigt), tSymGradientInertiaStress_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }

    // test skew gradient inertia stress
    //
    auto tSkwGradientInertiaStress_Host = Kokkos::create_mirror_view( tSkwGradientInertiaStresses );
    Kokkos::deep_copy( tSkwGradientInertiaStress_Host, tSkwGradientInertiaStresses );

    std::vector<std::vector<Plato::Scalar>> tSkwGradientInertiaStress_Gold = { 
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08}
    };

    for(int iCell=0; iCell<int(tSkwGradientInertiaStress_Gold.size()); iCell++){
      for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
        if(tSkwGradientInertiaStress_Gold[iCell][iSkw] == 0.0){
          TEST_ASSERT(fabs(tSkwGradientInertiaStress_Host(iCell,iPoint,iSkw)) < 1e-14);
        } else {
          TEST_FLOATING_EQUALITY(tSkwGradientInertiaStress_Host(iCell,iPoint,iSkw), tSkwGradientInertiaStress_Gold[iCell][iSkw], 1e-13);
        }
      }
    }

    // test symmetric free inertia stress
    //
    auto tSymFreeInertiaStress_Host = Kokkos::create_mirror_view( tSymFreeInertiaStresses );
    Kokkos::deep_copy( tSymFreeInertiaStress_Host, tSymFreeInertiaStresses );

    std::vector<std::vector<Plato::Scalar>> tSymFreeInertiaStress_Gold = { 
      {-0.3225, -0.15, 0.0225, 2.86875, 3.20625, 3.54375},
      { -0.258, -0.12,  0.018,   2.295,   2.565,   2.835},
      {-0.2365, -0.11, 0.0165, 2.10375, 2.35125, 2.59875},
      {-0.2795, -0.13, 0.0195, 2.48625, 2.77875, 3.07125},
      { -0.344, -0.16,  0.024,    3.06,    3.42,    3.78},
      {-0.3655, -0.17, 0.0255, 3.25125, 3.63375, 4.01625}
    };

    for(int iCell=0; iCell<int(tSymFreeInertiaStress_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymFreeInertiaStress_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymFreeInertiaStress_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymFreeInertiaStress_Host(iCell,iPoint,iVoigt), tSymFreeInertiaStress_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }

    // test skew free inertia stress
    //
    auto tSkwFreeInertiaStress_Host = Kokkos::create_mirror_view( tSkwFreeInertiaStresses );
    Kokkos::deep_copy( tSkwFreeInertiaStress_Host, tSkwFreeInertiaStresses );
    
    std::vector<std::vector<Plato::Scalar>> tSkwFreeInertiaStress_Gold = { 
      {0, 0, 0, -1.125e-08, -1.125e-08, -1.125e-08},
      {0, 0, 0,     -9e-09,     -9e-09,     -9e-09},
      {0, 0, 0,  -8.25e-09,  -8.25e-09,  -8.25e-09},
      {0, 0, 0,  -9.75e-09,  -9.75e-09,  -9.75e-09},
      {0, 0, 0,   -1.2e-08,   -1.2e-08,   -1.2e-08},
      {0, 0, 0, -1.275e-08, -1.275e-08, -1.275e-08}
    };

    for(int iCell=0; iCell<int(tSkwFreeInertiaStress_Gold.size()); iCell++){
      for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
        if(tSkwFreeInertiaStress_Gold[iCell][iSkw] == 0.0){
          TEST_ASSERT(fabs(tSkwFreeInertiaStress_Host(iCell,iPoint,iSkw)) < 1e-14);
        } else {
          TEST_FLOATING_EQUALITY(tSkwFreeInertiaStress_Host(iCell,iPoint,iSkw), tSkwFreeInertiaStress_Gold[iCell][iSkw], 1e-13);
        }
      }
    }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, ComputeExpressionInertiaKinetics)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    using EvalType = typename Plato::Hyperbolic::Evaluation<ElementType>::Residual;

    const int tNumCells = tMesh->NumElements();
    const auto tNumPoints = ElementType::getCubWeights().size();
    TEST_EQUALITY(tNumCells, 6);
    TEST_EQUALITY(tNumPoints, 4);

    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;

    std::vector<std::vector<Plato::Scalar>> tKnownControl = {
      {0.8, 1.0, 0.5, 0.7},
      {0.2, 0.4, 0.6, 0.8},
      {0.4, 0.05, 0.3, 0.25}};
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("density", tNumCells, ElementType::mNumNodesPerCell);
    Plato::TestHelpers::setControlWS(tKnownControl, tControlWS);

    std::vector<Plato::Scalar>
      tControlWeights = {0.861803398874990, 0.455278640450005, 0.160557280900008, 0.0, 0.0, 0.0};

    std::vector<Plato::Scalar> tKnownSymGradientMicroInertia = { 
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002,
      8e-05, 8e-05, 6e-05, 0.00016, 0.00026, 0.0002
    }; // tNumCells x tNumVoigtTerms
    auto tSymGradientMicroInertiaVals = Plato::TestHelpers::create_device_view(tKnownSymGradientMicroInertia);
    Plato::ScalarArray3DT<Plato::Scalar> tSymGradientMicroInertias("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymGradientMicroInertias(iCell,iPoint,iVoigt) = tSymGradientMicroInertiaVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwGradientMicroInertia = { 
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012,
      -8e-05, -0.00022, -0.00012
    }; // tNumCells x tNumSkwTerms // tNumCells x tNumSkwTerms
    auto tSkwGradientMicroInertiaVals = Plato::TestHelpers::create_device_view(tKnownSkwGradientMicroInertia);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwGradientMicroInertias("", tNumCells, tNumPoints, tNumSkwTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++)
          tSkwGradientMicroInertias(iCell,iPoint,iSkw) = tSkwGradientMicroInertiaVals(iCell*tNumSkwTerms+iSkw);
    });

    std::vector<Plato::Scalar> tKnownSymFreeMicroInertia = { 
      0.00015, 0.0001875, 0.000225, 0.0006375, 0.0007125, 0.0007875,
      0.00012,   0.00015,  0.00018,   0.00051,   0.00057,   0.00063,
      0.00011, 0.0001375, 0.000165, 0.0004675, 0.0005225, 0.0005775,
      0.00013, 0.0001625, 0.000195, 0.0005525, 0.0006175, 0.0006825,
      0.00016,    0.0002,  0.00024,   0.00068,   0.00076,   0.00084,
      0.00017, 0.0002125, 0.000255, 0.0007225, 0.0008075, 0.0008925
    }; // tNumCells x tNumVoigtTerms
    auto tSymFreeMicroInertiaVals = Plato::TestHelpers::create_device_view(tKnownSymFreeMicroInertia);
    Plato::ScalarArray3DT<Plato::Scalar> tSymFreeMicroInertias("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymFreeMicroInertias(iCell,iPoint,iVoigt) = tSymFreeMicroInertiaVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwFreeMicroInertia = { 
      -0.0001125, -0.0001125, -0.0001125,
          -9e-05,     -9e-05,     -9e-05,
       -8.25e-05,  -8.25e-05,  -8.25e-05,
       -9.75e-05,  -9.75e-05,  -9.75e-05,
        -0.00012,   -0.00012,   -0.00012,
      -0.0001275, -0.0001275, -0.0001275
    }; // tNumCells x tNumSkwTerms
    auto tSkwFreeMicroInertiaVals = Plato::TestHelpers::create_device_view(tKnownSkwFreeMicroInertia);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwFreeMicroInertias("", tNumCells, tNumPoints, tNumSkwTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iSkw=0; iSkw<tNumSkwTerms; iSkw++)
          tSkwFreeMicroInertias(iCell,iPoint,iSkw) = tSkwFreeMicroInertiaVals(iCell*tNumSkwTerms+iSkw);
    });

    auto tParams = setup_inertia_model_expression_parameter_list();
    Plato::Hyperbolic::Micromorphic::InertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");

    Plato::Hyperbolic::Micromorphic::MicromorphicKineticsFactory<EvalType, ElementType> tKineticsFactory;
    auto tKinetics = tKineticsFactory.create(tInertiaModel);

    Plato::ScalarArray3DT<Plato::Scalar> tSymGradientInertiaStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwGradientInertiaStresses("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSymFreeInertiaStresses    ("stress", tNumCells, tNumPoints, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwFreeInertiaStresses    ("stress", tNumCells, tNumPoints, tNumVoigtTerms);

    // Compute kinetics
    (*tKinetics)(tSymGradientInertiaStresses, tSkwGradientInertiaStresses, tSymFreeInertiaStresses, tSkwFreeInertiaStresses, tSymGradientMicroInertias, tSkwGradientMicroInertias, tSymFreeMicroInertias, tSkwFreeMicroInertias, tControlWS);

    constexpr unsigned int iPoint = 0;
    // test symmetric gradient inertia stress
    //
    auto tSymGradientInertiaStress_Host = Kokkos::create_mirror_view( tSymGradientInertiaStresses );
    Kokkos::deep_copy( tSymGradientInertiaStress_Host, tSymGradientInertiaStresses );

    std::vector<std::vector<Plato::Scalar>> tSymGradientInertiaStress_Gold = { 
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05},
      {0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05}
    };

    // add weighting due to expression: E0*(1+Z)
    for (int i=0; i<tSymGradientInertiaStress_Gold.size(); i++) {
      for (auto& tVal : tSymGradientInertiaStress_Gold[i]) {
        tVal *= (1.0+tControlWeights[i]);
      }
    }

    for(int iCell=0; iCell<int(tSymGradientInertiaStress_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymGradientInertiaStress_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymGradientInertiaStress_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymGradientInertiaStress_Host(iCell,iPoint,iVoigt), tSymGradientInertiaStress_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }

    // test skew gradient inertia stress
    //
    auto tSkwGradientInertiaStress_Host = Kokkos::create_mirror_view( tSkwGradientInertiaStresses );
    Kokkos::deep_copy( tSkwGradientInertiaStress_Host, tSkwGradientInertiaStresses );

    std::vector<std::vector<Plato::Scalar>> tSkwGradientInertiaStress_Gold = { 
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08},
      {0, 0, 0, -8e-09, -2.2e-08, -1.2e-08}
    };

    // add weighting due to expression: E0*(1+Z)
    for (int i=0; i<tSkwGradientInertiaStress_Gold.size(); i++) {
      for (auto& tVal : tSkwGradientInertiaStress_Gold[i]) {
        tVal *= (1.0+tControlWeights[i]);
      }
    }

    for(int iCell=0; iCell<int(tSkwGradientInertiaStress_Gold.size()); iCell++){
      for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
        if(tSkwGradientInertiaStress_Gold[iCell][iSkw] == 0.0){
          TEST_ASSERT(fabs(tSkwGradientInertiaStress_Host(iCell,iPoint,iSkw)) < 1e-14);
        } else {
          TEST_FLOATING_EQUALITY(tSkwGradientInertiaStress_Host(iCell,iPoint,iSkw), tSkwGradientInertiaStress_Gold[iCell][iSkw], 1e-13);
        }
      }
    }

    // test symmetric free inertia stress
    //
    auto tSymFreeInertiaStress_Host = Kokkos::create_mirror_view( tSymFreeInertiaStresses );
    Kokkos::deep_copy( tSymFreeInertiaStress_Host, tSymFreeInertiaStresses );

    std::vector<std::vector<Plato::Scalar>> tSymFreeInertiaStress_Gold = { 
      {-0.3225, -0.15, 0.0225, 2.86875, 3.20625, 3.54375},
      { -0.258, -0.12,  0.018,   2.295,   2.565,   2.835},
      {-0.2365, -0.11, 0.0165, 2.10375, 2.35125, 2.59875},
      {-0.2795, -0.13, 0.0195, 2.48625, 2.77875, 3.07125},
      { -0.344, -0.16,  0.024,    3.06,    3.42,    3.78},
      {-0.3655, -0.17, 0.0255, 3.25125, 3.63375, 4.01625}
    };

    // add weighting due to expression: E0*(1+Z)
    for (int i=0; i<tSymFreeInertiaStress_Gold.size(); i++) {
      for (auto& tVal : tSymFreeInertiaStress_Gold[i]) {
        tVal *= (1.0+tControlWeights[i]);
      }
    }

    for(int iCell=0; iCell<int(tSymFreeInertiaStress_Gold.size()); iCell++){
      for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++){
        if(tSymFreeInertiaStress_Gold[iCell][iVoigt] == 0.0){
          TEST_ASSERT(fabs(tSymFreeInertiaStress_Host(iCell,iPoint,iVoigt)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tSymFreeInertiaStress_Host(iCell,iPoint,iVoigt), tSymFreeInertiaStress_Gold[iCell][iVoigt], 1e-13);
        }
      }
    }

    // test skew free inertia stress
    //
    auto tSkwFreeInertiaStress_Host = Kokkos::create_mirror_view( tSkwFreeInertiaStresses );
    Kokkos::deep_copy( tSkwFreeInertiaStress_Host, tSkwFreeInertiaStresses );
    
    std::vector<std::vector<Plato::Scalar>> tSkwFreeInertiaStress_Gold = { 
      {0, 0, 0, -1.125e-08, -1.125e-08, -1.125e-08},
      {0, 0, 0,     -9e-09,     -9e-09,     -9e-09},
      {0, 0, 0,  -8.25e-09,  -8.25e-09,  -8.25e-09},
      {0, 0, 0,  -9.75e-09,  -9.75e-09,  -9.75e-09},
      {0, 0, 0,   -1.2e-08,   -1.2e-08,   -1.2e-08},
      {0, 0, 0, -1.275e-08, -1.275e-08, -1.275e-08}
    };

    // add weighting due to expression: E0*(1+Z)
    for (int i=0; i<tSkwFreeInertiaStress_Gold.size(); i++) {
      for (auto& tVal : tSkwFreeInertiaStress_Gold[i]) {
        tVal *= (1.0+tControlWeights[i]);
      }
    }
    for(int iCell=0; iCell<int(tSkwFreeInertiaStress_Gold.size()); iCell++){
      for(int iSkw=0; iSkw<tNumVoigtTerms; iSkw++){
        if(tSkwFreeInertiaStress_Gold[iCell][iSkw] == 0.0){
          TEST_ASSERT(fabs(tSkwFreeInertiaStress_Host(iCell,iPoint,iSkw)) < 1e-14);
        } else {
          TEST_FLOATING_EQUALITY(tSkwFreeInertiaStress_Host(iCell,iPoint,iSkw), tSkwFreeInertiaStress_Gold[iCell][iSkw], 1e-13);
        }
      }
    }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, FullStressDivergence)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;

    const int tNumCells = tMesh->NumElements();
    const auto tCubWeights = ElementType::getCubWeights();
    const auto tNumPoints = tCubWeights.size();
    constexpr int tNumDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;
    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;
                        
    std::vector<Plato::Scalar> tKnownVolumes = { 
      0.125, 0.125, 0.125, 
      0.125, 0.125, 0.125,
    }; // tNumCells x tNumPoints
    auto tVolumeVals = Plato::TestHelpers::create_device_view(tKnownVolumes);
    Plato::ScalarMultiVectorT<Plato::Scalar> tVolumes("", tNumCells, tNumPoints);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        tVolumes(iCell,iPoint) = tVolumeVals(iCell);
    });

    std::vector<Plato::Scalar> tKnownGradients = { 
      0.0, -2.0, 0.0,   2.0, 0.0, -2.0,   -2.0, 2.0, 0.0,   0.0, 0.0, 2.0,
      0.0, -2.0, 0.0,   0.0, 2.0, -2.0,   -2.0, 0.0, 2.0,   2.0, 0.0, 0.0,
      0.0, 0.0, -2.0,   -2.0, 2.0, 0.0,   0.0, -2.0, 2.0,   2.0, 0.0, 0.0,
      0.0, 0.0, -2.0,   -2.0, 0.0, 2.0,   2.0, -2.0, 0.0,   0.0, 2.0, 0.0,
      -2.0, 0.0, 0.0,   0.0, -2.0, 2.0,   2.0, 0.0, -2.0,   0.0, 2.0, 0.0,
      -2.0, 0.0, 0.0,   2.0, -2.0, 0.0,   0.0, 2.0, -2.0,   0.0, 0.0, 2.0
    }; // tNumCells x tNumNodesPerCell x tSpaceDim
    auto tGradientVals = Plato::TestHelpers::create_device_view(tKnownGradients);
    Plato::ScalarArray4DT<Plato::Scalar> tGradients("",tNumCells,tNumPoints,tNumNodesPerCell,tSpaceDim);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iNode=0; iNode<tNumNodesPerCell; iNode++)
          for(int iDim=0; iDim<tSpaceDim; iDim++)
            tGradients(iCell,iPoint,iNode,iDim) = tGradientVals(iCell*tNumNodesPerCell*tSpaceDim+iNode*tSpaceDim+iDim);
    });

    std::vector<Plato::Scalar> tKnownSymCauchyStress = { 
      -0.0003664195, -0.000784252, -0.0014249285, -3.996675e-05, -3.787425e-05, -4.917375e-05,
       -0.000167986, -0.000502252,  -0.001059362,   -2.9295e-05,   -2.5947e-05,   -3.5991e-05,
      -0.0001018415, -0.000408252, -0.0009375065, -2.573775e-05, -2.197125e-05, -3.159675e-05,
      -0.0002341305, -0.000596252, -0.0011812175, -3.285225e-05, -2.992275e-05, -4.038525e-05,
       -0.000432564, -0.000878252,  -0.001546784,   -4.3524e-05,    -4.185e-05,   -5.3568e-05,
      -0.0004987085, -0.000972252, -0.0016686395, -4.708125e-05, -4.582575e-05, -5.796225e-05,
    }; // tNumCells x tNumVoigtTerms
    auto tSymCauchyStressVals = Plato::TestHelpers::create_device_view(tKnownSymCauchyStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSymCauchyStresses("",tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymCauchyStresses(iCell,iPoint,iVoigt) = tSymCauchyStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwCauchyStress = { 
      0, 0, 0, 5.85000000000001e-11, -1.935e-10, -1.35000000000001e-11,
      0, 0, 0,              1.8e-11,  -2.34e-10, -5.40000000000001e-11,
      0, 0, 0, 4.50000000000003e-12, -2.475e-10,             -6.75e-11,
      0, 0, 0, 3.15000000000002e-11, -2.205e-10,             -4.05e-11,
      0, 0, 0, 7.20000000000001e-11,   -1.8e-10,                     0,
      0, 0, 0,             8.55e-11, -1.665e-10,  1.35000000000001e-11,
    }; // tNumCells x tNumVoigtTerms
    auto tSkwCauchyStressVals = Plato::TestHelpers::create_device_view(tKnownSkwCauchyStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwCauchyStresses("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSkwCauchyStresses(iCell,iPoint,iVoigt) = tSkwCauchyStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    Plato::ScalarMultiVectorT<Plato::Scalar> tResidual("residual",tNumCells,tNumDofsPerCell);

    Plato::Hyperbolic::Micromorphic::FullStressDivergence<ElementType> computeFullStressDivergence;

    Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
    {
        tVolumes(cellOrdinal,gpOrdinal) *= tCubWeights(gpOrdinal);
        computeFullStressDivergence(cellOrdinal,gpOrdinal,tResidual,tSymCauchyStresses,tSkwCauchyStresses,tGradients,tVolumes);
    });

  // test residual
  //
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  // just testing the first 3 elements since there is a large number of values
  std::vector<std::vector<Plato::Scalar>> tResidual_gold = { 
   {     2.0489068125e-06,  3.26771666666666e-05,      1.6652836875e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -1.36893773541667e-05, -3.83626874999999e-07,  5.77939351458332e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     1.32185723541667e-05, -3.06282609791666e-05, -8.71979999999996e-08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -1.5781018125e-06,     -1.6652788125e-06, -5.93720208333332e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},

   {       1.49962725e-06,  2.09271666666666e-05,        1.22062575e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -4.18492499999999e-07, -1.97065424166666e-05,  4.29194575833333e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     5.91828191666666e-06,          2.789985e-07, -4.30589680833333e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -6.99941666666666e-06,       -1.49962275e-06,       -1.08111525e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},

   { 9.15479062499999e-07,   1.0724060625e-06,  3.90627708333333e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     2.92686177083333e-06, -1.56939715625e-05,          -1.56948e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     4.01054999999999e-07,  1.59380939375e-05, -3.79903643958333e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -4.24339583333333e-06,  -1.3165284375e-06, -9.15458437499999e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
  };

  for(int iCell=0; iCell<int(tResidual_gold.size()); iCell++){
    for(int iDof=0; iDof<tNumDofsPerCell; iDof++){
      if(tResidual_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(iCell,iDof), tResidual_gold[iCell][iDof], 1e-13);
      }
    }
  }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, FullStressDivergence_InertiaStresses)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;

    const int tNumCells = tMesh->NumElements();
    const auto tCubWeights = ElementType::getCubWeights();
    const auto tNumPoints = tCubWeights.size();
    constexpr int tNumDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr int tNumNodesPerCell = ElementType::mNumNodesPerCell;
    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;
                        
    std::vector<Plato::Scalar> tKnownVolumes = { 
      0.125, 0.125, 0.125, 
      0.125, 0.125, 0.125,
    }; // tNumCells x tNumPoints
    auto tVolumeVals = Plato::TestHelpers::create_device_view(tKnownVolumes);
    Plato::ScalarMultiVectorT<Plato::Scalar> tVolumes("", tNumCells, tNumPoints);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        tVolumes(iCell,iPoint) = tVolumeVals(iCell);
    });

    std::vector<Plato::Scalar> tKnownGradients = { 
      0.0, -2.0, 0.0,   2.0, 0.0, -2.0,   -2.0, 2.0, 0.0,   0.0, 0.0, 2.0,
      0.0, -2.0, 0.0,   0.0, 2.0, -2.0,   -2.0, 0.0, 2.0,   2.0, 0.0, 0.0,
      0.0, 0.0, -2.0,   -2.0, 2.0, 0.0,   0.0, -2.0, 2.0,   2.0, 0.0, 0.0,
      0.0, 0.0, -2.0,   -2.0, 0.0, 2.0,   2.0, -2.0, 0.0,   0.0, 2.0, 0.0,
      -2.0, 0.0, 0.0,   0.0, -2.0, 2.0,   2.0, 0.0, -2.0,   0.0, 2.0, 0.0,
      -2.0, 0.0, 0.0,   2.0, -2.0, 0.0,   0.0, 2.0, -2.0,   0.0, 0.0, 2.0
    }; // tNumCells x tNumNodesPerCell x tSpaceDim
    auto tGradientVals = Plato::TestHelpers::create_device_view(tKnownGradients);
    Plato::ScalarArray4DT<Plato::Scalar> tGradients("",tNumCells,tNumPoints,tNumNodesPerCell,tSpaceDim);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iNode=0; iNode<tNumNodesPerCell; iNode++)
          for(int iDim=0; iDim<tSpaceDim; iDim++)
            tGradients(iCell,iPoint,iNode,iDim) = tGradientVals(iCell*tNumNodesPerCell*tSpaceDim+iNode*tSpaceDim+iDim);
    });

    std::vector<Plato::Scalar> tKnownSymGradientInertiaStress = { 
      0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05,
      0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05,
      0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05,
      0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05,
      0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05,
      0.000536, 0.000536, 0.000512, 3.2e-05, 5.2e-05, 4e-05
    }; // tNumCells x tNumVoigtTerms
    auto tSymGradientInertiaStressVals = Plato::TestHelpers::create_device_view(tKnownSymGradientInertiaStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSymGradientInertiaStresses("",tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymGradientInertiaStresses(iCell,iPoint,iVoigt) = tSymGradientInertiaStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwGradientInertiaStress = { 
      0, 0, 0, -8e-09, -2.2e-08, -1.2e-08,
      0, 0, 0, -8e-09, -2.2e-08, -1.2e-08,
      0, 0, 0, -8e-09, -2.2e-08, -1.2e-08,
      0, 0, 0, -8e-09, -2.2e-08, -1.2e-08,
      0, 0, 0, -8e-09, -2.2e-08, -1.2e-08,
      0, 0, 0, -8e-09, -2.2e-08, -1.2e-08
    }; // tNumCells x tNumVoigtTerms
    auto tSkwGradientInertiaStressVals = Plato::TestHelpers::create_device_view(tKnownSkwGradientInertiaStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwGradientInertiaStresses("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSkwGradientInertiaStresses(iCell,iPoint,iVoigt) = tSkwGradientInertiaStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    Plato::ScalarMultiVectorT<Plato::Scalar> tResidual("residual",tNumCells,tNumDofsPerCell);

    Plato::Hyperbolic::Micromorphic::FullStressDivergence<ElementType> computeFullStressDivergence;

    Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
    {
        tVolumes(cellOrdinal,gpOrdinal) *= tCubWeights(gpOrdinal);
        computeFullStressDivergence(cellOrdinal,gpOrdinal,tResidual,tSymGradientInertiaStresses,tSkwGradientInertiaStresses,tGradients,tVolumes);
    });

  // test residual
  //
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  // just testing the first 3 elements since there is a large number of values
  std::vector<std::vector<Plato::Scalar>> tResidual_gold = { 
   {-1.66616666666666e-06, -2.23333333333333e-05, -1.33366666666666e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     2.01675833333333e-05,  3.34166666666666e-07,         -1.916575e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -2.06671666666666e-05,  2.06661666666666e-05, -8.33916666666666e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
              2.16575e-06,             1.333e-06,  2.13333333333333e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},

   {-1.66616666666666e-06, -2.23333333333333e-05, -1.33366666666667e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -4.99583333333332e-07,  2.10003333333333e-05, -1.99996666666666e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -2.01675833333333e-05, -3.34166666666666e-07,          1.916575e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     2.23333333333333e-05,  1.66716666666666e-06,  2.16758333333333e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},

   {         -2.16575e-06,            -1.333e-06, -2.13333333333333e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    -2.06671666666666e-05,  2.06661666666666e-05, -8.33916666666665e-07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     4.99583333333332e-07, -2.10003333333333e-05,  1.99996666666666e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
     2.23333333333333e-05,  1.66716666666666e-06,  2.16758333333333e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
  };

  for(int iCell=0; iCell<int(tResidual_gold.size()); iCell++){
    for(int iDof=0; iDof<tNumDofsPerCell; iDof++){
      if(tResidual_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(iCell,iDof), tResidual_gold[iCell][iDof], 1e-13);
      }
    }
  }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, ProjectStressToNode)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;

    const int tNumCells = tMesh->NumElements();
    const auto tCubPoints = ElementType::getCubPoints();
    const auto tCubWeights = ElementType::getCubWeights();
    const auto tNumPoints = tCubWeights.size();
    constexpr int tNumDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;
                        
    std::vector<Plato::Scalar> tKnownVolumes = { 
      0.125, 0.125, 0.125, 
      0.125, 0.125, 0.125,
    }; // tNumCells x tNumPoints
    auto tVolumeVals = Plato::TestHelpers::create_device_view(tKnownVolumes);
    Plato::ScalarMultiVectorT<Plato::Scalar> tVolumes("", tNumCells, tNumPoints);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        tVolumes(iCell,iPoint) = tVolumeVals(iCell);
    });

    std::vector<Plato::Scalar> tKnownSymCauchyStress = { 
      -0.0003664195, -0.000784252, -0.0014249285, -3.996675e-05, -3.787425e-05, -4.917375e-05,
       -0.000167986, -0.000502252,  -0.001059362,   -2.9295e-05,   -2.5947e-05,   -3.5991e-05,
      -0.0001018415, -0.000408252, -0.0009375065, -2.573775e-05, -2.197125e-05, -3.159675e-05,
      -0.0002341305, -0.000596252, -0.0011812175, -3.285225e-05, -2.992275e-05, -4.038525e-05,
       -0.000432564, -0.000878252,  -0.001546784,   -4.3524e-05,    -4.185e-05,   -5.3568e-05,
      -0.0004987085, -0.000972252, -0.0016686395, -4.708125e-05, -4.582575e-05, -5.796225e-05,
    }; // tNumCells x tNumVoigtTerms
    auto tSymCauchyStressVals = Plato::TestHelpers::create_device_view(tKnownSymCauchyStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSymCauchyStresses("",tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymCauchyStresses(iCell,iPoint,iVoigt) = tSymCauchyStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwCauchyStress = { 
      0, 0, 0, 5.85000000000001e-11, -1.935e-10, -1.35000000000001e-11,
      0, 0, 0,              1.8e-11,  -2.34e-10, -5.40000000000001e-11,
      0, 0, 0, 4.50000000000003e-12, -2.475e-10,             -6.75e-11,
      0, 0, 0, 3.15000000000002e-11, -2.205e-10,             -4.05e-11,
      0, 0, 0, 7.20000000000001e-11,   -1.8e-10,                     0,
      0, 0, 0,             8.55e-11, -1.665e-10,  1.35000000000001e-11,
    }; // tNumCells x tNumVoigtTerms
    auto tSkwCauchyStressVals = Plato::TestHelpers::create_device_view(tKnownSkwCauchyStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwCauchyStresses("", tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSkwCauchyStresses(iCell,iPoint,iVoigt) = tSkwCauchyStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSymMicroStress = { 
      0.00178317375, 0.00197495625, 0.00216673875,  0.00115566,  0.00129162,  0.00142758,
        0.001426539,   0.001579965,   0.001733391, 0.000924528, 0.001033296, 0.001142064,
      0.00130766075, 0.00144830125, 0.00158894175, 0.000847484, 0.000947188, 0.001046892,
      0.00154541725, 0.00171162875, 0.00187784025, 0.001001572, 0.001119404, 0.001237236,
        0.001902052,    0.00210662,   0.002311188, 0.001232704, 0.001377728, 0.001522752,
      0.00202093025, 0.00223828375, 0.00245563725, 0.001309748, 0.001463836, 0.001617924,
    }; // tNumCells x tNumVoigtTerms
    auto tSymMicroStressVals = Plato::TestHelpers::create_device_view(tKnownSymMicroStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSymMicroStresses("",tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymMicroStresses(iCell,iPoint,iVoigt) = tSymMicroStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    Plato::ScalarMultiVectorT<Plato::Scalar> tResidual("residual",tNumCells,tNumDofsPerCell);

    Plato::Hyperbolic::Micromorphic::ProjectStressToNode<ElementType, tSpaceDim> computeStressForMicromorphicResidual;

    Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
    {
        auto tCubPoint = tCubPoints(gpOrdinal);
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        tVolumes(cellOrdinal,gpOrdinal) *= tCubWeights(gpOrdinal);
        computeStressForMicromorphicResidual(cellOrdinal,gpOrdinal,tResidual,tSymCauchyStresses,tSkwCauchyStresses,tSymMicroStresses,tBasisValues,tVolumes);
    });

  // test residual
  //
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  // just testing the first 2 elements since there is a large number of values
  std::vector<std::vector<Plato::Scalar>> tResidual_gold = { 
   {0.0, 0.0, 0.0, 1.11957981770833e-05, 1.43708763020833e-05, 1.87066002604166e-05, 6.22722235156249e-06, 6.92445022656249e-06, 7.69142585156249e-06, 6.22722296093749e-06, 6.92444821093749e-06, 7.69142571093749e-06,
    0.0, 0.0, 0.0, 1.11957981770833e-05, 1.43708763020833e-05, 1.87066002604166e-05, 6.22722235156249e-06, 6.92445022656249e-06, 7.69142585156249e-06, 6.22722296093749e-06, 6.92444821093749e-06, 7.69142571093749e-06,
    0.0, 0.0, 0.0, 1.11957981770833e-05, 1.43708763020833e-05, 1.87066002604166e-05, 6.22722235156249e-06, 6.92445022656249e-06, 7.69142585156249e-06, 6.22722296093749e-06, 6.92444821093749e-06, 7.69142571093749e-06,
    0.0, 0.0, 0.0, 1.11957981770833e-05, 1.43708763020833e-05, 1.87066002604166e-05, 6.22722235156249e-06, 6.92445022656249e-06, 7.69142585156249e-06, 6.22722296093749e-06, 6.92444821093749e-06, 7.69142571093749e-06},

   {0.0, 0.0, 0.0, 8.30481770833332e-06, 1.08448802083333e-05, 1.45455885416666e-05, 4.96782803124999e-06, 5.51689184374999e-06, 6.13570340624999e-06, 4.96782821874999e-06, 5.51688940624999e-06, 6.13570284374999e-06,
    0.0, 0.0, 0.0, 8.30481770833332e-06, 1.08448802083333e-05, 1.45455885416666e-05, 4.96782803124999e-06, 5.51689184374999e-06, 6.13570340624999e-06, 4.96782821874999e-06, 5.51688940624999e-06, 6.13570284374999e-06,
    0.0, 0.0, 0.0, 8.30481770833332e-06, 1.08448802083333e-05, 1.45455885416666e-05, 4.96782803124999e-06, 5.51689184374999e-06, 6.13570340624999e-06, 4.96782821874999e-06, 5.51688940624999e-06, 6.13570284374999e-06,
    0.0, 0.0, 0.0, 8.30481770833332e-06, 1.08448802083333e-05, 1.45455885416666e-05, 4.96782803124999e-06, 5.51689184374999e-06, 6.13570340624999e-06, 4.96782821874999e-06, 5.51688940624999e-06, 6.13570284374999e-06},

   {0.0, 0.0, 0.0, 7.34115755208332e-06, 9.66954817708332e-06, 1.31585846354166e-05, 4.54802992447916e-06, 5.04770571614583e-06, 5.61712925781249e-06, 4.54802997135416e-06, 5.04770313802083e-06, 5.61712855468749e-06,
    0.0, 0.0, 0.0, 7.34115755208332e-06, 9.66954817708332e-06, 1.31585846354166e-05, 4.54802992447916e-06, 5.04770571614583e-06, 5.61712925781249e-06, 4.54802997135416e-06, 5.04770313802083e-06, 5.61712855468749e-06,
    0.0, 0.0, 0.0, 7.34115755208332e-06, 9.66954817708332e-06, 1.31585846354166e-05, 4.54802992447916e-06, 5.04770571614583e-06, 5.61712925781249e-06, 4.54802997135416e-06, 5.04770313802083e-06, 5.61712855468749e-06,
    0.0, 0.0, 0.0, 7.34115755208332e-06, 9.66954817708332e-06, 1.31585846354166e-05, 4.54802992447916e-06, 5.04770571614583e-06, 5.61712925781249e-06, 4.54802997135416e-06, 5.04770313802083e-06, 5.61712855468749e-06}
  };

  for(int iCell=0; iCell<int(tResidual_gold.size()); iCell++){
    for(int iDof=0; iDof<tNumDofsPerCell; iDof++){
      if(tResidual_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(iCell,iDof), tResidual_gold[iCell][iDof], 1e-13);
      }
    }
  }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, ProjectStressToNode_InertiaStresses)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;

    const int tNumCells = tMesh->NumElements();
    const auto tCubPoints = ElementType::getCubPoints();
    const auto tCubWeights = ElementType::getCubWeights();
    const auto tNumPoints = tCubWeights.size();
    constexpr int tNumDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr int tNumVoigtTerms = ElementType::mNumVoigtTerms;
    constexpr int tNumSkwTerms = ElementType::mNumSkwTerms;
                        
    std::vector<Plato::Scalar> tKnownVolumes = { 
      0.125, 0.125, 0.125, 
      0.125, 0.125, 0.125,
    }; // tNumCells x tNumPoints
    auto tVolumeVals = Plato::TestHelpers::create_device_view(tKnownVolumes);
    Plato::ScalarMultiVectorT<Plato::Scalar> tVolumes("", tNumCells, tNumPoints);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        tVolumes(iCell,iPoint) = tVolumeVals(iCell);
    });

    std::vector<Plato::Scalar> tKnownSymFreeInertiaStress = { 
      -0.3225, -0.15, 0.0225, 2.86875, 3.20625, 3.54375,
       -0.258, -0.12,  0.018,   2.295,   2.565,   2.835,
      -0.2365, -0.11, 0.0165, 2.10375, 2.35125, 2.59875,
      -0.2795, -0.13, 0.0195, 2.48625, 2.77875, 3.07125,
       -0.344, -0.16,  0.024,    3.06,    3.42,    3.78,
      -0.3655, -0.17, 0.0255, 3.25125, 3.63375, 4.01625
    }; // tNumCells x tNumVoigtTerms
    auto tSymFreeInertiaStressVals = Plato::TestHelpers::create_device_view(tKnownSymFreeInertiaStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSymFreeInertiaStresses("",tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSymFreeInertiaStresses(iCell,iPoint,iVoigt) = tSymFreeInertiaStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    std::vector<Plato::Scalar> tKnownSkwFreeInertiaStress = { 
      0, 0, 0, -1.125e-08, -1.125e-08, -1.125e-08,
      0, 0, 0,     -9e-09,     -9e-09,     -9e-09,
      0, 0, 0,  -8.25e-09,  -8.25e-09,  -8.25e-09,
      0, 0, 0,  -9.75e-09,  -9.75e-09,  -9.75e-09,
      0, 0, 0,   -1.2e-08,   -1.2e-08,   -1.2e-08,
      0, 0, 0, -1.275e-08, -1.275e-08, -1.275e-08
    }; // tNumCells x tNumVoigtTerms
    auto tSkwFreeInertiaStressVals = Plato::TestHelpers::create_device_view(tKnownSkwFreeInertiaStress);
    Plato::ScalarArray3DT<Plato::Scalar> tSkwFreeInertiaStresses("",tNumCells, tNumPoints, tNumVoigtTerms);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        for(int iVoigt=0; iVoigt<tNumVoigtTerms; iVoigt++)
          tSkwFreeInertiaStresses(iCell,iPoint,iVoigt) = tSkwFreeInertiaStressVals(iCell*tNumVoigtTerms+iVoigt);
    });

    Plato::ScalarMultiVectorT<Plato::Scalar> tResidual("residual",tNumCells,tNumDofsPerCell);

    Plato::Hyperbolic::Micromorphic::ProjectStressToNode<ElementType, tSpaceDim> computeStressForMicromorphicResidual;

    Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
    {
        auto tCubPoint = tCubPoints(gpOrdinal);
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        tVolumes(cellOrdinal,gpOrdinal) *= tCubWeights(gpOrdinal);
        computeStressForMicromorphicResidual(cellOrdinal,gpOrdinal,tResidual,tSymFreeInertiaStresses,tSkwFreeInertiaStresses,tBasisValues,tVolumes);
    });

  // test residual
  //
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  // just testing the first 2 elements since there is a large number of values
  std::vector<std::vector<Plato::Scalar>> tResidual_gold = { 
   {0.0, 0.0, 0.0, -0.0016796875, -0.000781249999999999, 0.0001171875, 0.0149414061914062, 0.0166992186914062, 0.0184570311914062, 0.0149414063085937, 0.0166992188085937, 0.0184570313085937,
    0.0, 0.0, 0.0, -0.0016796875, -0.000781249999999999, 0.0001171875, 0.0149414061914062, 0.0166992186914062, 0.0184570311914062, 0.0149414063085937, 0.0166992188085937, 0.0184570313085937,
    0.0, 0.0, 0.0, -0.0016796875, -0.000781249999999999, 0.0001171875, 0.0149414061914062, 0.0166992186914062, 0.0184570311914062, 0.0149414063085937, 0.0166992188085937, 0.0184570313085937,
    0.0, 0.0, 0.0, -0.0016796875, -0.000781249999999999, 0.0001171875, 0.0149414061914062, 0.0166992186914062, 0.0184570311914062, 0.0149414063085937, 0.0166992188085937, 0.0184570313085937},

   {0.0, 0.0, 0.0, -0.00134375, -0.000624999999999999, 9.37499999999996e-05, 0.011953124953125, 0.013359374953125, 0.014765624953125, 0.011953125046875, 0.013359375046875, 0.014765625046875,
   0.0, 0.0, 0.0, -0.00134375, -0.000624999999999999, 9.37499999999996e-05, 0.011953124953125, 0.013359374953125, 0.014765624953125, 0.011953125046875, 0.013359375046875, 0.014765625046875,
   0.0, 0.0, 0.0, -0.00134375, -0.000624999999999999, 9.37499999999996e-05, 0.011953124953125, 0.013359374953125, 0.014765624953125, 0.011953125046875, 0.013359375046875, 0.014765625046875,
   0.0, 0.0, 0.0, -0.00134375, -0.000624999999999999, 9.37499999999996e-05, 0.011953124953125, 0.013359374953125, 0.014765624953125, 0.011953125046875, 0.013359375046875, 0.014765625046875},

   {0.0, 0.0, 0.0, -0.00123177083333333, -0.000572916666666665, 8.59374999999994e-05, 0.0109570312070312, 0.0122460937070312, 0.0135351562070312, 0.0109570312929687, 0.0122460937929687, 0.0135351562929687,
   0.0, 0.0, 0.0, -0.00123177083333333, -0.000572916666666665, 8.59374999999994e-05, 0.0109570312070312, 0.0122460937070312, 0.0135351562070312, 0.0109570312929687, 0.0122460937929687, 0.0135351562929687,
   0.0, 0.0, 0.0, -0.00123177083333333, -0.000572916666666665, 8.59374999999994e-05, 0.0109570312070312, 0.0122460937070312, 0.0135351562070312, 0.0109570312929687, 0.0122460937929687, 0.0135351562929687,
   0.0, 0.0, 0.0, -0.00123177083333333, -0.000572916666666665, 8.59374999999994e-05, 0.0109570312070312, 0.0122460937070312, 0.0135351562070312, 0.0109570312929687, 0.0122460937929687, 0.0135351562929687}
  };

  for(int iCell=0; iCell<int(tResidual_gold.size()); iCell++){
    for(int iDof=0; iDof<tNumDofsPerCell; iDof++){
      if(tResidual_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(tResidual_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tResidual_Host(iCell,iDof), tResidual_gold[iCell][iDof], 1e-13);
      }
    }
  }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicElementFunctorTests, InertiaContribution)
{
    constexpr int tMeshWidth=1;
    constexpr int tSpaceDim=3;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;

    const int tNumCells = tMesh->NumElements();
    const int tNumNodes = tMesh->NumNodes();
    const auto tCubPoints = ElementType::getCubPoints();
    const auto tCubWeights = ElementType::getCubWeights();
    const auto tNumPoints = tCubWeights.size();
    constexpr int tNumDofsPerCell = ElementType::mNumDofsPerCell;
    constexpr int tNumDofsPerNode = ElementType::mNumDofsPerNode;

    int tNumDofs = tNumNodes*tNumDofsPerNode;
    Plato::ScalarVector tStateDotDot("state dot dot", tNumDofs);
    Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
    {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tStateDotDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-5)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
    });
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateDotDotWS("state dot dot workset",tNumCells, tNumDofsPerCell);
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);
    tWorksetBase.worksetState(tStateDotDot, tStateDotDotWS);

    std::vector<Plato::Scalar> tKnownVolumes = { 
      0.125, 0.125, 0.125, 
      0.125, 0.125, 0.125,
    }; // tNumCells x tNumPoints
    auto tVolumeVals = Plato::TestHelpers::create_device_view(tKnownVolumes);
    Plato::ScalarMultiVectorT<Plato::Scalar> tVolumes("", tNumCells, tNumPoints);
    Kokkos::parallel_for("populate", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int iCell, const int iPoint)
    {
        tVolumes(iCell,iPoint) = tVolumeVals(iCell);
    });

    auto tParams = setup_inertia_model_expression_parameter_list();
    Plato::Hyperbolic::Micromorphic::InertiaModelFactory<tSpaceDim> tInertiaModelFactory(*tParams);
    auto tInertiaModel = tInertiaModelFactory.create("material_1");
    
    Plato::InterpolateFromNodal<ElementType, tNumDofsPerNode, /*offset=*/0, tSpaceDim> interpolateFromNodal;
    Plato::InertialContent<ElementType> computeInertialContent(tInertiaModel);
    Plato::ProjectToNode<ElementType, tSpaceDim> projectInertialContent;

    Plato::ScalarMultiVectorT<Plato::Scalar> tResidual("residual",tNumCells,tNumDofsPerCell);

    Kokkos::parallel_for("gradients", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {tNumCells, tNumPoints}),
    KOKKOS_LAMBDA(const int cellOrdinal, const int gpOrdinal)
    {
        auto tCubPoint = tCubPoints(gpOrdinal);
        auto tBasisValues = ElementType::basisValues(tCubPoint);
        tVolumes(cellOrdinal,gpOrdinal) *= tCubWeights(gpOrdinal);

        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tAcceleration(0.0);
        Plato::Array<ElementType::mNumSpatialDims, Plato::Scalar> tInertialContent(0.0);

        interpolateFromNodal(cellOrdinal, tBasisValues, tStateDotDotWS, tAcceleration);
        computeInertialContent(tInertialContent, tAcceleration);
        projectInertialContent(cellOrdinal, tVolumes(cellOrdinal,gpOrdinal), tBasisValues, tInertialContent, tResidual);
    });

    // test residual
    //
    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    // just testing the first 3 elements since there is a large number of values
    std::vector<std::vector<Plato::Scalar>>
      tResidual_gold = {
        {0.226843749999999e-3, 0.453687499999998e-3, 0.680531249999997e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.317581250000001e-3, 0.635162500000002e-3, 0.952743750000003e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.257089583333334e-3, 0.514179166666669e-3, 0.771268750000003e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.332704166666668e-3, 0.665408333333336e-3, 0.998112500000003e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.181474999999999e-3, 0.362949999999998e-3, 0.544424999999997e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.211720833333334e-3, 0.423441666666668e-3, 0.635162500000003e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.226843750000001e-3, 0.453687500000002e-3, 0.680531250000003e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.287335416666668e-3, 0.574670833333335e-3, 0.862006250000003e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.166352083333333e-3, 0.332704166666665e-3, 0.499056249999998e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.211720833333334e-3, 0.423441666666668e-3, 0.635162500000003e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.181475000000001e-3, 0.362950000000002e-3, 0.544425000000002e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.272212500000001e-3, 0.544425000000002e-3, 0.816637500000002e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
      };

    for(int iCell=0; iCell<int(tResidual_gold.size()); iCell++){
      for(int iDof=0; iDof<tNumDofsPerCell; iDof++){
        if(tResidual_gold[iCell][iDof] == 0.0){
          TEST_ASSERT(fabs(tResidual_Host(iCell,iDof)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(tResidual_Host(iCell,iDof), tResidual_gold[iCell][iDof], 1e-13);
        }
      }
    }

}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicResidualTests, ErrorAFormNotSpecified)
{
    // set parameters
    //
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "            <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "            <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <ParameterList  name='Te Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='0.6'/> <!-- Eta_bar_1 -->  \n"
      "            <Parameter  name='Lambda' type='double' value='2.0'/>  <!-- Eta_bar_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='0.2'/>  <!-- Eta_bar_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Tc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_bar_2 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jm Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='2300.0'/>  <!-- Eta_1 --> \n"
      "            <Parameter  name='Lambda' type='double' value='-1800.0'/>  <!-- Eta_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='4500.0'/>  <!-- Eta_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_2 --> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.0'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
    );

    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParams, tDataMap);
    TEST_THROW(
    Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::MicromorphicMechanics<Plato::Tet4>>
      tVectorFunction(tSpatialModel, tDataMap, *tParams, tParams->get<std::string>("PDE Constraint")),
    std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicResidualTests, ErrorAFormFalse)
{
    // set parameters
    //
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "            <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "            <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <ParameterList  name='Te Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='0.6'/> <!-- Eta_bar_1 -->  \n"
      "            <Parameter  name='Lambda' type='double' value='2.0'/>  <!-- Eta_bar_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='0.2'/>  <!-- Eta_bar_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Tc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_bar_2 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jm Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='2300.0'/>  <!-- Eta_1 --> \n"
      "            <Parameter  name='Lambda' type='double' value='-1800.0'/>  <!-- Eta_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='4500.0'/>  <!-- Eta_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_2 --> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='A-Form' type='bool' value='false'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.0'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
    );

    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParams, tDataMap);
    TEST_THROW(
    Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::MicromorphicMechanics<Plato::Tet4>>
      tVectorFunction(tSpatialModel, tDataMap, *tParams, tParams->get<std::string>("PDE Constraint")),
    std::runtime_error);
}

TEUCHOS_UNIT_TEST(RelaxedMicromorphicResidualTests, ErrorExplicitNotSpecified)
{
    // set parameters
    //
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "  <ParameterList name='Plato Problem'>                                    \n"
      "    <Parameter name='Physics' type='string' value='Micromorphic Mechanical' />  \n"
      "    <Parameter name='PDE Constraint' type='string' value='Hyperbolic' /> \n"
      "    <ParameterList name='Material Models'>                           \n"
      "      <ParameterList name='material_1'>                           \n"
      "        <ParameterList name='Cubic Micromorphic Linear Elastic'>     \n"
      "          <ParameterList  name='Ce Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='-120.74'/>   \n"
      "            <Parameter  name='Mu' type='double' value='557.11'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='8.37'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cc Stiffness Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.8e-4'/>   \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Cm Stiffness Tensor'>   \n"
      "            <Parameter  name='Lambda' type='double' value='180.63'/>   \n"
      "            <Parameter  name='Mu' type='double' value='255.71'/>   \n"
      "            <Parameter  name='Alpha' type='double' value='181.28'/>   \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "        <ParameterList name='Cubic Micromorphic Inertia'>     \n"
      "          <Parameter  name='Mass Density' type='double' value='1451.8'/>   \n"
      "          <ParameterList  name='Te Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='0.6'/> <!-- Eta_bar_1 -->  \n"
      "            <Parameter  name='Lambda' type='double' value='2.0'/>  <!-- Eta_bar_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='0.2'/>  <!-- Eta_bar_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Tc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_bar_2 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jm Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='2300.0'/>  <!-- Eta_1 --> \n"
      "            <Parameter  name='Lambda' type='double' value='-1800.0'/>  <!-- Eta_3 --> \n"
      "            <Parameter  name='Alpha' type='double' value='4500.0'/>  <!-- Eta_star_1 --> \n"
      "          </ParameterList>                                                  \n"
      "          <ParameterList  name='Jc Inertia Tensor'>   \n"
      "            <Parameter  name='Mu' type='double' value='1.0e-4'/>  <!-- Eta_2 --> \n"
      "          </ParameterList>                                                  \n"
      "        </ParameterList>                                                  \n"
      "      </ParameterList>                                                  \n"
      "    </ParameterList>                                                  \n"
      "    <ParameterList name='Spatial Model'>                                    \n"
      "      <ParameterList name='Domains'>                                        \n"
      "        <ParameterList name='Design Volume'>                                \n"
      "          <Parameter name='Element Block' type='string' value='body'/>      \n"
      "          <Parameter name='Material Model' type='string' value='material_1'/> \n"
      "        </ParameterList>                                                    \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Hyperbolic'>                                    \n"
      "      <ParameterList name='Penalty Function'>                                    \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>      \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>      \n"
      "        <Parameter name='Minimum Value' type='double' value='1e-9'/>      \n"
      "      </ParameterList>                                                      \n"
      "    </ParameterList>                                                        \n"
      "    <ParameterList name='Time Integration'>                                    \n"
      "      <Parameter name='Termination Time' type='double' value='20.0e-6'/>      \n"
      "      <Parameter name='A-Form' type='bool' value='true'/>      \n"
      "      <Parameter name='Newmark Gamma' type='double' value='0.5'/>      \n"
      "      <Parameter name='Newmark Beta' type='double' value='0.25'/>      \n"
      "    </ParameterList>                                                        \n"
      "  </ParameterList>                                                  \n"
    );

    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, *tParams, tDataMap);
    TEST_THROW(
    Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::MicromorphicMechanics<Plato::Tet4>>
      tVectorFunction(tSpatialModel, tDataMap, *tParams, tParams->get<std::string>("PDE Constraint")),
    std::runtime_error);
}

TEUCHOS_UNIT_TEST( RelaxedMicromorphicResidualTests, 3D_NoInertia )
{
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    const auto tCubWeights = ElementType::getCubWeights();
    const auto tNumPoints = tCubWeights.size();

    auto tAllParams = setup_full_model_parameter_list_no_inertia();
    auto tParams = tAllParams->sublist("Plato Problem");

    // create test mesh
    //
    constexpr int cMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    int tNumNodes = tMesh->NumNodes();
    int tNumDofs = tNumNodes*tNumDofsPerNode;

    // create vector function
    //
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, tParams, tDataMap);

    Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::MicromorphicMechanics<Plato::Tet4>>
      tVectorFunction(tSpatialModel, tDataMap, tParams, tParams.get<std::string>("PDE Constraint"));

    // create control
    //
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    
    // create mesh based state
    //
    Plato::ScalarVector tState("state", tNumDofs);
    Plato::ScalarVector tStateDot("state dot", tNumDofs);
    Plato::ScalarVector tStateDotDot("state dot dot", tNumDofs);
    Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
    {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tState(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-7)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-6)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDotDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-5)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
    });

    // test residual
    //
    auto tResidual = tVectorFunction.value(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar>
      tNode0Residual_gold = {
        5.98970098750004e-05, 0.000126645364500000, 0.000206365402625000, 7.11731624999997e-05, 0.000118481437499999, 0.000165789712499999,
        5.18153740999998e-05, 5.70897079749997e-05, 6.36195080999997e-05, 5.18153758999997e-05, 5.70896670249997e-05, 6.36194918999997e-05,
      };
    
    int tNodeOrdinal = 0;
    for(int iDof=0; iDof<tNumDofsPerNode; iDof++){
        int tLocalOrdinal = tNodeOrdinal*tNumDofsPerNode + iDof;
        TEST_FLOATING_EQUALITY(tResidual_Host[tLocalOrdinal], tNode0Residual_gold[iDof], 1e-12);
    }

    // test gradient wrt U
    //
    auto tJacobian = tVectorFunction.gradient_u(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jac_entries = tJacobian->entries();
    auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
    Kokkos::deep_copy(jac_entriesHost, jac_entries);

    std::vector<Plato::Scalar>
      gold_jac_entries = {
          168.370060000000,                 0,                 0,  20.6974999999999, -2.51541666666665, -2.51541666666665,                 0, 0.174378749999999, 0.174378749999999,                 0, 0.174371249999999, 0.174371249999999,
                         0,  168.370060000000,                 0, -2.51541666666665,  20.6974999999999, -2.51541666666665, 0.174378749999999,                 0, 0.174371249999999, 0.174371249999999,                 0, 0.174378749999999,
                         0,                 0,  168.370060000000, -2.51541666666665, -2.51541666666665,  20.6974999999999, 0.174371249999999, 0.174371249999999,                 0, 0.174378749999999, 0.174378749999999,                 0,
          20.6974999999999, -2.51541666666665, -2.51541666666665,  21.0691249999998, 0.748624999999995, 0.748624999999995,                 0,                 0,                 0,                 0,                 0,                 0,
         -2.51541666666665,  20.6974999999999, -2.51541666666665, 0.748624999999995,  21.0691249999998, 0.748624999999995,                 0,                 0,                 0,                 0,                 0,                 0,
         -2.51541666666665, -2.51541666666665,  20.6974999999999, 0.748624999999995, 0.748624999999995,  21.0691249999998,                 0,                 0,                 0,                 0,                 0,                 0,
                         0, 0.174378749999999, 0.174371249999999,                 0,                 0,                 0,  2.37062724999998,                 0,                 0,  2.37062274999998,                 0,                 0,
         0.174378749999999,                 0, 0.174371249999999,                 0,                 0,                 0,                 0,  2.37062724999998,                 0,                 0,  2.37062274999998,                 0,
         0.174378749999999, 0.174371249999999,                 0,                 0,                 0,                 0,                 0,                 0,  2.37062724999998,                 0,                 0,  2.37062274999998,
                         0, 0.174371249999999, 0.174378749999999,                 0,                 0,                 0,  2.37062274999998,                 0,                 0,  2.37062724999998,                 0,                 0,
         0.174371249999999,                 0, 0.174378749999999,                 0,                 0,                 0,                 0,  2.37062274999998,                 0,                 0,  2.37062724999998,                 0,
         0.174371249999999, 0.174378749999999,                 0,                 0,                 0,                 0,                 0,                 0,  2.37062274999998,                 0,                 0,  2.37062724999998
      };

    int jac_entriesSize = gold_jac_entries.size();
    for(int i=0; i<jac_entriesSize; i++){
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-12);
    }

    // test gradient wrt V
    //
    auto tJacobianV = tVectorFunction.gradient_v(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacV_entries = tJacobianV->entries();
    auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
    Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

    std::vector<Plato::Scalar> gold_jacV_entries = {
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int jacV_entriesSize = gold_jacV_entries.size();
    for(int i=0; i<jacV_entriesSize; i++){
      if(gold_jacV_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacV_entriesHost(i), gold_jacV_entries[i], 1.0e-12);
      }
    }

    // test gradient wrt A
    //
    auto tJacobianA = tVectorFunction.gradient_a(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacA_entries = tJacobianA->entries();
    auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
    Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

    std::vector<Plato::Scalar> gold_jacA_entries = {
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int jacA_entriesSize = gold_jacA_entries.size();
    for(int i=0; i<jacA_entriesSize; i++){
      if(gold_jacA_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-12);
      }
    }
}

TEUCHOS_UNIT_TEST( RelaxedMicromorphicResidualTests, 3D_WithInertia )
{
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    const auto tCubWeights = ElementType::getCubWeights();
    const auto tNumPoints = tCubWeights.size();

    auto tAllParams = setup_full_model_parameter_list();
    auto tParams = tAllParams->sublist("Plato Problem");

    // create test mesh
    //
    constexpr int cMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    int tNumNodes = tMesh->NumNodes();
    int tNumDofs = tNumNodes*tNumDofsPerNode;

    // create vector function
    //
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, tParams, tDataMap);

    Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::MicromorphicMechanics<Plato::Tet4>>
      tVectorFunction(tSpatialModel, tDataMap, tParams, tParams.get<std::string>("PDE Constraint"));

    // create control
    //
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    
    // create mesh based state
    //
    Plato::ScalarVector tState("state", tNumDofs);
    Plato::ScalarVector tStateDot("state dot", tNumDofs);
    Plato::ScalarVector tStateDotDot("state dot dot", tNumDofs);
    Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
    {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tState(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-7)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-6)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDotDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-5)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
    });

    // test residual
    //
    auto tResidual = tVectorFunction.value(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar>
      tNode0Residual_gold = {
        0.00232474550987499, 0.00476232736449998, 0.00720488490262497, -0.0139038268374999, -0.00638151856249997, 0.00114078971250000,
          0.124364314886599,   0.138994589220474,   0.153626119020599,   0.124364315863399,    0.138994590154524,   0.153626119979399
    };

    int tNodeOrdinal = 0;
    for(int iDof=0; iDof<tNumDofsPerNode; iDof++){
        int tLocalOrdinal = tNodeOrdinal*tNumDofsPerNode + iDof;
        TEST_FLOATING_EQUALITY(tResidual_Host[tLocalOrdinal], tNode0Residual_gold[iDof], 1e-12);
    }

    // test gradient wrt U
    //
    auto tJacobian = tVectorFunction.gradient_u(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jac_entries = tJacobian->entries();
    auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
    Kokkos::deep_copy(jac_entriesHost, jac_entries);

    std::vector<Plato::Scalar>
      gold_jac_entries = {
         168.370060000000,                 0,                 0,  20.6974999999999, -2.51541666666665, -2.51541666666665,                 0, 0.174378749999999, 0.174378749999999,                 0, 0.174371249999999, 0.174371249999999,
                        0,  168.370060000000,                 0, -2.51541666666665,  20.6974999999999, -2.51541666666665, 0.174378749999999,                 0, 0.174371249999999, 0.174371249999999,                 0, 0.174378749999999,
                        0,                 0,  168.370060000000, -2.51541666666665, -2.51541666666665,  20.6974999999999, 0.174371249999999, 0.174371249999999,                 0, 0.174378749999999, 0.174378749999999,                 0,
         20.6974999999999, -2.51541666666665, -2.51541666666665,  21.0691249999998, 0.748624999999995, 0.748624999999995,                 0,                 0,                 0,                 0,                 0,                 0,
        -2.51541666666665,  20.6974999999999, -2.51541666666665, 0.748624999999995,  21.0691249999998, 0.748624999999995,                 0,                 0,                 0,                 0,                 0,                 0,
        -2.51541666666665, -2.51541666666665,  20.6974999999999, 0.748624999999995, 0.748624999999995,  21.0691249999998,                 0,                 0,                 0,                 0,                 0,                 0,
                        0, 0.174378749999999, 0.174371249999999,                 0,                 0,                 0,  2.37062724999998,                 0,                 0,  2.37062274999998,                 0,                 0,
        0.174378749999999,	               0, 0.174371249999999,                 0,                 0,                 0,                 0,  2.37062724999998,                 0,                 0,  2.37062274999998,                 0,
        0.174378749999999, 0.174371249999999,                 0,                 0,                 0,                 0,                 0,                 0,  2.37062724999998,                 0,                 0,  2.37062274999998,
                        0, 0.174371249999999, 0.174378749999999,                 0,                 0,                 0,  2.37062274999998,                 0,                 0,  2.37062724999998,                 0,                 0,
        0.174371249999999,                 0, 0.174378749999999,                 0,                 0,                 0,                 0,  2.37062274999998,                 0,                 0,  2.37062724999998,                 0,
        0.174371249999999, 0.174378749999999,                 0,                 0,                 0,                 0,                 0,                 0,  2.37062274999998,                 0,                 0,  2.37062724999998
      };

    int jac_entriesSize = gold_jac_entries.size();
    for(int i=0; i<jac_entriesSize; i++){
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-12);
    }

    // test gradient wrt V
    //
    auto tJacobianV = tVectorFunction.gradient_v(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacV_entries = tJacobianV->entries();
    auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
    Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

    std::vector<Plato::Scalar> gold_jacV_entries = {
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int jacV_entriesSize = gold_jacV_entries.size();
    for(int i=0; i<jacV_entriesSize; i++){
      if(gold_jacV_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacV_entriesHost(i), gold_jacV_entries[i], 1.0e-12);
      }
    }

    // test gradient wrt A
    //
    auto tJacobianA = tVectorFunction.gradient_a(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacA_entries = tJacobianA->entries();
    auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
    Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

    std::vector<Plato::Scalar>
      gold_jacA_entries = {
        18.7475333333332,                0,                0,                 0,                 0,                 0,                0,                0,                0,                0,                0,                0,
                       0, 18.7475333333332,                0,                 0,                 0,                 0,                0,                0,                0,                0,                0,                0,
                       0,                0, 18.7475333333332,                 0,                 0,                 0,                0,                0,                0,                0,                0,                0,
                       0,                0,                0,  34.9999999999998, -22.4999999999998, -22.4999999999998,                0,                0,                0,                0,                0,                0,
                       0,                0,                0, -22.4999999999998,  34.9999999999998, -22.4999999999998,                0,                0,                0,                0,                0,                0,
                       0,                0,                0, -22.4999999999998, -22.4999999999998,  34.9999999999998,                0,                0,                0,                0,                0,                0,
                       0,                0,                0,                 0,                 0,                 0, 56.2500012499996,                0,                0, 56.2499987499996,                0,                0,
                       0,                0,                0,                 0,                 0,                 0,                0, 56.2500012499996,                0,                0, 56.2499987499996,                0,
                       0,                0,                0,                 0,                 0,                 0,                0,                0, 56.2500012499996,                0,                0, 56.2499987499996,
                       0,                0,                0,                 0,                 0,                 0, 56.2499987499996,                0,                0, 56.2500012499996,                0,                0,
                       0,                0,                0,                 0,                 0,                 0,                0, 56.2499987499996,                0,                0, 56.2500012499996,                0,
                       0,                0,                0,                 0,                 0,                 0,                0,                0, 56.2499987499996,                0,                0, 56.2500012499996
      };

    int jacA_entriesSize = gold_jacA_entries.size();
    for(int i=0; i<jacA_entriesSize; i++){
      if(gold_jacA_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-12);
      }
    }
}

TEUCHOS_UNIT_TEST( RelaxedMicromorphicResidualTests, 3D_Expression_WithInertia_Density0p5 )
{
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    const auto tCubWeights = ElementType::getCubWeights();
    const auto tNumPoints = tCubWeights.size();

    auto tAllParams = setup_full_model_expression_parameter_list();
    auto tParams = tAllParams->sublist("Plato Problem");

    // create test mesh
    //
    constexpr int cMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", cMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Tet4>;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    int tNumNodes = tMesh->NumNodes();
    int tNumDofs = tNumNodes*tNumDofsPerNode;

    // create vector function
    //
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, tParams, tDataMap);

    Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::MicromorphicMechanics<Plato::Tet4>>
      tVectorFunction(tSpatialModel, tDataMap, tParams, tParams.get<std::string>("PDE Constraint"));

    // create control
    //
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    
    // create mesh based state
    //
    Plato::ScalarVector tState("state", tNumDofs);
    Plato::ScalarVector tStateDot("state dot", tNumDofs);
    Plato::ScalarVector tStateDotDot("state dot dot", tNumDofs);
    Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
    {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tState(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-7)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-6)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDotDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-5)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
    });

    // test residual
    //
    auto tResidual = tVectorFunction.value(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar>
      tNode0Residual_gold = {
        0.00230753076481249, 0.00478431604674998, 0.00726856485393747, -0.0208557402562499, -0.00957227784374996, 0.00171118456874999,
        0.186546472329899, 0.208491883830712, 0.230439178530899, 0.186546473795099, 0.208491885231787, 0.230439179969099
      };

    int tNodeOrdinal = 0;
    for(int iDof=0; iDof<tNumDofsPerNode; iDof++){
        int tLocalOrdinal = tNodeOrdinal*tNumDofsPerNode + iDof;
        TEST_FLOATING_EQUALITY(tResidual_Host[tLocalOrdinal], tNode0Residual_gold[iDof], 1e-12);
    }

    // test gradient wrt U
    //
    auto tJacobian = tVectorFunction.gradient_u(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jac_entries = tJacobian->entries();
    auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
    Kokkos::deep_copy(jac_entriesHost, jac_entries);

    std::vector<Plato::Scalar>
      gold_jac_entries = {
        252.555090000000,                 0,                 0,  31.0462499999998, -3.77312499999998, -3.77312499999998,                 0, 0.261568124999999, 0.261568124999999,                 0, 0.261556874999998, 0.261556874999998,
                       0,  252.555090000000,                 0, -3.77312499999998,  31.0462499999998, -3.77312499999998, 0.261568124999999,                 0, 0.261556874999998, 0.261556874999998,                 0, 0.261568124999999,
                       0,                 0,  252.555090000000, -3.77312499999998, -3.77312499999998,  31.0462499999998, 0.261556874999998, 0.261556874999998,                 0, 0.261568124999999, 0.261568124999999,                 0,
        31.0462499999998, -3.77312499999998, -3.77312499999998,  31.6036874999998,  1.12293749999999,  1.12293749999999,                 0,                 0,                 0,                 0,                 0,                 0,
       -3.77312499999998,  31.0462499999998, -3.77312499999998,  1.12293749999999,  31.6036874999998,  1.12293749999999,                 0,                 0,                 0,                 0,                 0,                 0,
       -3.77312499999998, -3.77312499999998,  31.0462499999998,  1.12293749999999,  1.12293749999999,  31.6036874999998,                 0,                 0,                 0,                 0,                 0,                 0,
                       0, 0.261568124999999, 0.261556874999998,                 0,                 0,                 0,  3.55594087499998,                 0,                 0,  3.55593412499998,                 0,                 0,
       0.261568124999999,                 0, 0.261556874999998,                 0,                 0,                 0,                 0,  3.55594087499998,                 0,                 0,  3.55593412499998,                 0,
       0.261568124999999, 0.261556874999998,                 0,                 0,                 0,                 0,                 0,                 0,  3.55594087499998,                 0,                 0,  3.55593412499998,
                       0, 0.261556874999998, 0.261568124999999,                 0,                 0,                 0,  3.55593412499998,                 0,                 0,  3.55594087499998,                 0,                 0,
       0.261556874999998,                 0, 0.261568124999999,                 0,                 0,                 0,                 0,  3.55593412499998,                 0,                 0,  3.55594087499998,                 0,
       0.261556874999998, 0.261568124999999,                 0,                 0,                 0,                 0,                 0,                 0,  3.55593412499998,                 0,                 0,  3.55594087499998
      };

    int jac_entriesSize = gold_jac_entries.size();
    for(int i=0; i<jac_entriesSize; i++){
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-12);
    }

    // test gradient wrt V
    //
    auto tJacobianV = tVectorFunction.gradient_v(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacV_entries = tJacobianV->entries();
    auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
    Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

    std::vector<Plato::Scalar> gold_jacV_entries = {
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    int jacV_entriesSize = gold_jacV_entries.size();
    for(int i=0; i<jacV_entriesSize; i++){
      if(gold_jacV_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacV_entriesHost(i), 1.0 * gold_jacV_entries[i], 1.0e-12);
      }
    }

    // test gradient wrt A
    //
    auto tJacobianA = tVectorFunction.gradient_a(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacA_entries = tJacobianA->entries();
    auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
    Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

    std::vector<Plato::Scalar>
      gold_jacA_entries = {
        19.0475499999999,                0,                0,                 0,                 0,                 0,                0,                0,                0,                0,                0,                0,
                       0, 19.0475499999999,                0,                 0,                 0,                 0,                0,                0,                0,                0,                0,                0,
                       0,                0, 19.0475499999999,                 0,                 0,                 0,                0,                0,                0,                0,                0,                0,
                       0,                0,                0,  52.4999999999996, -33.7499999999998, -33.7499999999998,                0,                0,                0,                0,                0,                0,
                       0,                0,                0, -33.7499999999998,  52.4999999999996, -33.7499999999998,                0,                0,                0,                0,                0,                0,
                       0,                0,                0, -33.7499999999998, -33.7499999999998,  52.4999999999996,                0,                0,                0,                0,                0,                0,
                       0,                0,                0,                 0,                 0,                 0, 84.3750018749994,                0,                0, 84.3749981249994,                0,                0,
                       0,                0,                0,                 0,                 0,                 0,                0, 84.3750018749994,                0,                0, 84.3749981249994,                0,
                       0,                0,                0,                 0,                 0,                 0,                0,                0, 84.3750018749994,                0,                0, 84.3749981249994,
                       0,                0,                0,                 0,                 0,                 0, 84.3749981249994,                0,                0, 84.3750018749994,                0,                0,
                       0,                0,                0,                 0,                 0,                 0,                0, 84.3749981249994,                0,                0, 84.3750018749994,                0,
                       0,                0,                0,                 0,                 0,                 0,                0,                0, 84.3749981249994,                0,                0, 84.3750018749994
      };

    int jacA_entriesSize = gold_jacA_entries.size();
    for(int i=0; i<jacA_entriesSize; i++){
      if(gold_jacA_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-12);
      }
    }
}

#ifdef PLATO_HEX_ELEMENTS
TEUCHOS_UNIT_TEST( RelaxedMicromorphicResidualTests, 2D_Quad4_NoInertia )
{
    auto tAllParams = setup_full_model_parameter_list_no_inertia();
    auto tParams = tAllParams->sublist("Plato Problem");

    // create test mesh
    //
    constexpr int cMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("QUAD4", cMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Quad4>;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    int tNumNodes = tMesh->NumNodes();
    int tNumDofs = tNumNodes*tNumDofsPerNode;

    // create vector function
    //
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, tParams, tDataMap);

    Plato::Hyperbolic::VectorFunction<Plato::Hyperbolic::MicromorphicMechanics<Plato::Quad4>>
      tVectorFunction(tSpatialModel, tDataMap, tParams, tParams.get<std::string>("PDE Constraint"));

    // create control
    //
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    
    // create mesh based state
    //
    Plato::ScalarVector tState("state", tNumDofs);
    Plato::ScalarVector tStateDot("state dot", tNumDofs);
    Plato::ScalarVector tStateDotDot("state dot dot", tNumDofs);
    Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
    {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tState(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-7)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-6)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDotDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-5)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
    });

    // test residual
    //
    auto tResidual = tVectorFunction.value(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar> tNode0Residual_gold = {
      -2.19570034166667e-05, 5.549208825e-05, 9.89758333333334e-06, 3.73723333333333e-05, 1.66522180833333e-05, 1.66521985833333e-05
    };
    int tNodeOrdinal = 0;
    for(int iDof=0; iDof<tNumDofsPerNode; iDof++){
        int tLocalOrdinal = tNodeOrdinal*tNumDofsPerNode + iDof;
        TEST_FLOATING_EQUALITY(tResidual_Host[tLocalOrdinal], tNode0Residual_gold[iDof], 1e-12);
    }

    // test gradient wrt U
    //
    auto tJacobian = tVectorFunction.gradient_u(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jac_entries = tJacobian->entries();
    auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
    Kokkos::deep_copy(jac_entriesHost, jac_entries);

    std::vector<Plato::Scalar> gold_jac_entries = {
                 333.95006,        -28.092545,            82.79, -10.0616666666667,         0.697515,         0.697485,
                -28.092545,         333.95006,-10.0616666666667,             82.79,         0.697485,         0.697515,
                     82.79, -10.0616666666667, 46.8202777777778,  1.66361111111111,                0,                0,
         -10.0616666666667,             82.79, 1.66361111111111,  46.8202777777778,                0,                0,
                  0.697515,          0.697485,                0,                 0, 5.26806055555556, 5.26805055555556,
                  0.697485,          0.697515,                0,                 0, 5.26805055555556, 5.26806055555556
    };

    int jac_entriesSize = gold_jac_entries.size();
    for(int i=0; i<jac_entriesSize; i++){
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-12);
    }

    // test gradient wrt V
    //
    auto tJacobianV = tVectorFunction.gradient_v(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacV_entries = tJacobianV->entries();
    auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
    Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

    std::vector<Plato::Scalar> gold_jacV_entries = {
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
    };

    int jacV_entriesSize = gold_jacV_entries.size();
    for(int i=0; i<jacV_entriesSize; i++){
      if(gold_jacV_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacV_entriesHost(i), 1.0 * gold_jacV_entries[i], 1.0e-12);
      }
    }

    // test gradient wrt A
    //
    auto tJacobianA = tVectorFunction.gradient_a(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacA_entries = tJacobianA->entries();
    auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
    Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

    std::vector<Plato::Scalar> gold_jacA_entries = {
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
    };

    int jacA_entriesSize = gold_jacA_entries.size();
    for(int i=0; i<jacA_entriesSize; i++){
      if(gold_jacA_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-12);
      }
    }
}

TEUCHOS_UNIT_TEST( RelaxedMicromorphicResidualTests, 2D_Quad4_WithInertia )
{
    auto tAllParams = setup_full_model_parameter_list();
    auto tParams = tAllParams->sublist("Plato Problem");

    // create test mesh
    //
    constexpr int cMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("QUAD4", cMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Quad4>;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    int tNumNodes = tMesh->NumNodes();
    int tNumDofs = tNumNodes*tNumDofsPerNode;

    // create vector function
    //
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, tParams, tDataMap);

    Plato::Hyperbolic::VectorFunction<Plato::Hyperbolic::MicromorphicMechanics<Plato::Quad4>>
      tVectorFunction(tSpatialModel, tDataMap, tParams, tParams.get<std::string>("PDE Constraint"));

    // create control
    //
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    
    // create mesh based state
    //
    Plato::ScalarVector tState("state", tNumDofs);
    Plato::ScalarVector tStateDot("state dot", tNumDofs);
    Plato::ScalarVector tStateDotDot("state dot dot", tNumDofs);
    Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
    {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tState(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-7)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-6)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDotDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-5)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
    });

    // test residual
    //
    auto tResidual = tVectorFunction.value(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar> tNode0Residual_gold = {
      0.00111287882991667, 0.00240615625491667, 0.00100989758333333, 0.00487070566666667, 0.04126665213475, 0.0412666522819167
    };
    int tNodeOrdinal = 0;
    for(int iDof=0; iDof<tNumDofsPerNode; iDof++){
        int tLocalOrdinal = tNodeOrdinal*tNumDofsPerNode + iDof;
        TEST_FLOATING_EQUALITY(tResidual_Host[tLocalOrdinal], tNode0Residual_gold[iDof], 1e-12);
    }

    // test gradient wrt U
    //
    auto tJacobian = tVectorFunction.gradient_u(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jac_entries = tJacobian->entries();
    auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
    Kokkos::deep_copy(jac_entriesHost, jac_entries);

    std::vector<Plato::Scalar> gold_jac_entries = {
                 333.95006,        -28.092545,            82.79, -10.0616666666667,         0.697515,         0.697485,
                -28.092545,         333.95006,-10.0616666666667,             82.79,         0.697485,         0.697515,
                     82.79, -10.0616666666667, 46.8202777777778,  1.66361111111111,                0,                0,
         -10.0616666666667,             82.79, 1.66361111111111,  46.8202777777778,                0,                0,
                  0.697515,          0.697485,                0,                 0, 5.26806055555556, 5.26805055555556,
                  0.697485,          0.697515,                0,                 0, 5.26805055555556, 5.26806055555556
    };

    int jac_entriesSize = gold_jac_entries.size();
    for(int i=0; i<jac_entriesSize; i++){
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-12);
    }

    // test gradient wrt V
    //
    auto tJacobianV = tVectorFunction.gradient_v(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacV_entries = tJacobianV->entries();
    auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
    Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

    std::vector<Plato::Scalar> gold_jacV_entries = {
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
    };

    int jacV_entriesSize = gold_jacV_entries.size();
    for(int i=0; i<jacV_entriesSize; i++){
      if(gold_jacV_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacV_entriesHost(i), 1.0 * gold_jacV_entries[i], 1.0e-12);
      }
    }

    // test gradient wrt A
    //
    auto tJacobianA = tVectorFunction.gradient_a(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacA_entries = tJacobianA->entries();
    auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
    Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

    std::vector<Plato::Scalar> gold_jacA_entries = {
     41.4611444444444,         0.549975,                0,                0,                0,                0,
             0.549975, 41.4611444444444,                0,                0,                0,                0,
                    0,                0, 77.7777777777778,              -50,                0,                0,
                    0,                0,              -50, 77.7777777777778,                0,                0,
                    0,                0,                0,                0, 125.000002777778, 124.999997222222,
                    0,                0,                0,                0, 124.999997222222, 125.000002777778
    };

    int jacA_entriesSize = gold_jacA_entries.size();
    for(int i=0; i<jacA_entriesSize; i++){
      if(gold_jacA_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-12);
      }
    }
}

TEUCHOS_UNIT_TEST( RelaxedMicromorphicResidualTests, 2D_Quad4_Expression_WithInertia_Density0p5 )
{
    auto tAllParams = setup_full_model_expression_parameter_list();
    auto tParams = tAllParams->sublist("Plato Problem");

    // create test mesh
    //
    constexpr int cMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("QUAD4", cMeshWidth);
    using ElementType = typename Plato::Hyperbolic::MicromorphicMechanicsElement<Plato::Quad4>;
    constexpr int tNumDofsPerNode  = ElementType::mNumDofsPerNode;
    int tNumNodes = tMesh->NumNodes();
    int tNumDofs = tNumNodes*tNumDofsPerNode;

    // create vector function
    //
    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, tParams, tDataMap);

    Plato::Hyperbolic::VectorFunction<Plato::Hyperbolic::MicromorphicMechanics<Plato::Quad4>>
      tVectorFunction(tSpatialModel, tDataMap, tParams, tParams.get<std::string>("PDE Constraint"));

    // create control
    //
    auto tNumVerts = tMesh->NumNodes();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(0.5, tControl);
    
    // create mesh based state
    //
    Plato::ScalarVector tState("state", tNumDofs);
    Plato::ScalarVector tStateDot("state dot", tNumDofs);
    Plato::ScalarVector tStateDotDot("state dot dot", tNumDofs);
    Kokkos::parallel_for("state", Kokkos::RangePolicy<int>(0,tNumNodes), KOKKOS_LAMBDA(const int & aNodeOrdinal)
    {
      for (int tDofOrdinal=0; tDofOrdinal<tNumDofsPerNode; tDofOrdinal++)
      {
          tState(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-7)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-6)*(tDofOrdinal + 1)*aNodeOrdinal;
          tStateDotDot(aNodeOrdinal*tNumDofsPerNode+tDofOrdinal) = (1e-5)*(tDofOrdinal + 1)*aNodeOrdinal;
      }
    });

    // test residual
    //
    auto tResidual = tVectorFunction.value(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
    Kokkos::deep_copy( tResidual_Host, tResidual );

    std::vector<Plato::Scalar> tNode0Residual_gold = {
      0.00106440157820833, 0.00239940104904167, 0.001514846375, 0.0073060585, 0.061899978202125, 0.061899978422875
    };
    int tNodeOrdinal = 0;
    for(int iDof=0; iDof<tNumDofsPerNode; iDof++){
        int tLocalOrdinal = tNodeOrdinal*tNumDofsPerNode + iDof;
        TEST_FLOATING_EQUALITY(tResidual_Host[tLocalOrdinal], tNode0Residual_gold[iDof], 1e-12);
    }

    // test gradient wrt U
    //
    auto tJacobian = tVectorFunction.gradient_u(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jac_entries = tJacobian->entries();
    auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
    Kokkos::deep_copy(jac_entriesHost, jac_entries);

    std::vector<Plato::Scalar> gold_jac_entries = {
                 500.92509, -42.1388175,          124.185,         -15.0925,        1.0462725,        1.0462275,
               -42.1388175,   500.92509,         -15.0925,          124.185,        1.0462275,        1.0462725,
                   124.185,    -15.0925, 70.2304166666667, 2.49541666666667,                0,                0,
                  -15.0925,     124.185, 2.49541666666667, 70.2304166666667,                0,                0,
                 1.0462725,   1.0462275,                0,                0, 7.90209083333333, 7.90207583333334,
                 1.0462275,   1.0462725,                0,                0, 7.90207583333334, 7.90209083333333
    };

    int jac_entriesSize = gold_jac_entries.size();
    for(int i=0; i<jac_entriesSize; i++){
      TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-12);
    }

    // test gradient wrt V
    //
    auto tJacobianV = tVectorFunction.gradient_v(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacV_entries = tJacobianV->entries();
    auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
    Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

    std::vector<Plato::Scalar> gold_jacV_entries = {
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0,
    };

    int jacV_entriesSize = gold_jacV_entries.size();
    for(int i=0; i<jacV_entriesSize; i++){
      if(gold_jacV_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacV_entriesHost(i), 1.0 * gold_jacV_entries[i], 1.0e-12);
      }
    }

    // test gradient wrt A
    //
    auto tJacobianA = tVectorFunction.gradient_a(tState, tStateDot, tStateDotDot, tControl, 0.0);
    auto jacA_entries = tJacobianA->entries();
    auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
    Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

    std::vector<Plato::Scalar> gold_jacA_entries = {

     42.0278277777778,        0.8249625,                0,                0,                0,                0,
            0.8249625, 42.0278277777778,                0,                0,                0,                0,
                    0,                0, 116.666666666667,              -75,                0,                0,
                    0,                0,              -75, 116.666666666667,                0,                0,
                    0,                0,                0,                0, 187.500004166667, 187.499995833333,
                    0,                0,                0,                0, 187.499995833333, 187.500004166667
    };

    int jacA_entriesSize = gold_jacA_entries.size();
    for(int i=0; i<jacA_entriesSize; i++){
      if(gold_jacA_entries[i] == 0.0){
        TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-12);
      }
    }
}
#endif

TEUCHOS_UNIT_TEST(RelaxedMicromorphicProblemTests, ConstructWithFactory)
{
    // create test mesh
    //
    constexpr int tMeshWidth=2;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);
    int tNumCells = tMesh->NumElements();

    auto tParams = setup_full_model_parameter_list();

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    Plato::ProblemFactory tProblemFactory;
    TEST_NOTHROW(tProblemFactory.create(tMesh, *tParams, tMachine));
}

}
