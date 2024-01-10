#include "NaturalBCs.hpp"
#include "util/PlatoTestHelpers.hpp"
#include "SpatialModel.hpp"
#include "MechanicsElement.hpp"
#include "Tet4.hpp"
#include "Hex8.hpp"
#include "WorksetBase.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

namespace
{
Teuchos::RCP<Teuchos::ParameterList> pressureBCParameters(const std::string& aBCXML)
{
    constexpr auto kProblemXML = 
        "<ParameterList name='Plato Problem'>\n"
        "  <ParameterList name='Spatial Model'>\n"
        "    <ParameterList name='Domains'>\n"
        "      <ParameterList name='Design Volume'>\n"
        "        <Parameter name='Element Block' type='string' value='block_1'/>\n"
        "        <Parameter name='Material Model' type='string' value='Flubber'/>\n"
        "      </ParameterList>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n"
        "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>\n"
        "  <Parameter name='Self-Adjoint' type='bool' value='true'/>\n"
        "  <ParameterList name='Elliptic'>\n"
        "    <ParameterList name='Penalty Function'>\n"
        "      <Parameter name='Type' type='string' value='SIMP'/>\n"
        "      <Parameter name='Exponent' type='double' value='1.0'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n"
        "  <ParameterList name='Material Models'>\n"
        "    <ParameterList name='Flubber'>\n"
        "      <ParameterList name='Isotropic Linear Elastic'>\n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>\n"
        "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>\n"
        "      </ParameterList>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n";
    constexpr auto kClosingXML = "</ParameterList>\n";
    return Teuchos::getParametersFromXmlString(std::string(kProblemXML) + aBCXML + std::string(kClosingXML));
}

template<typename BCType>
void testBCDataConstruction(
    Teuchos::ParameterList& aBCParameters,
    Teuchos::FancyOStream &aOut, 
    bool &aSuccess)
{
    using ElementType = Plato::MechanicsElement<Plato::Tet4>;
    bool tCtorSuccess = true;
    TEUCHOS_TEST_NOTHROW(Plato::NaturalBCs<ElementType>(aBCParameters.sublist("Natural Boundary Conditions")), aOut, tCtorSuccess);
    Plato::NaturalBCs<ElementType> tTestBC(aBCParameters.sublist("Natural Boundary Conditions"));

    bool tSizeSuccess = true;
    TEUCHOS_TEST_EQUALITY(tTestBC.numNaturalBCs(), 1, aOut, tSizeSuccess);

    bool tCastSuccess = true;
    TEUCHOS_TEST_NOTHROW(dynamic_cast<const BCType&>(tTestBC.getNaturalBC(0).getNaturalBCData()), aOut, tCastSuccess);

    aSuccess = tCtorSuccess && tSizeSuccess && tCastSuccess;
}

Plato::Scalar surfaceIntegralSum(Teuchos::ParameterList aInputParams)
{
    constexpr auto kMeshName = "nodal_surface_pressure_field.exo";
    Plato::Mesh tMesh = Plato::MeshFactory::create(kMeshName);

    using ElementType = Plato::MechanicsElement<Plato::Hex8>;
    Plato::WorksetBase<ElementType> tWorksetBase(tMesh);

    Plato::DataMap tDataMap;
    Plato::SpatialModel tSpatialModel(tMesh, aInputParams, tDataMap);

    const Plato::OrdinalType tNumCells = tMesh->NumElements();
    constexpr Plato::OrdinalType tNumNodesPerCell = ElementType::mNumNodesPerCell;
    constexpr Plato::OrdinalType tNumSpatialDims = ElementType::mNumSpatialDims;
    constexpr Plato::OrdinalType tNumGlobalDofsPerCell = ElementType::mNumDofsPerCell;

    Plato::ScalarArray3DT<Plato::Scalar> tConfig("Configuration Workset", tNumCells, tNumNodesPerCell, tNumSpatialDims);
    tWorksetBase.worksetConfig(tConfig);

    const Plato::ScalarMultiVectorT<Plato::Scalar> tState("State Workset", tNumCells, tNumGlobalDofsPerCell);
    const Plato::ScalarMultiVectorT<Plato::Scalar> tControl("Control Workset", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVectorT<Plato::Scalar> tResult("Result Workset", tNumCells, tNumGlobalDofsPerCell);

    Plato::NaturalBCs<ElementType> tTestBC(aInputParams.sublist("Natural Boundary Conditions"));

    constexpr Plato::Scalar kScale = 1.0;
    constexpr Plato::Scalar kTime = 0.0;
    tTestBC.get(tSpatialModel, tState, tControl, tConfig, tResult, kScale, kTime);

    Plato::Scalar tSumResult = 0.0;
    Kokkos::parallel_reduce(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{tNumCells, tNumGlobalDofsPerCell}),
    KOKKOS_LAMBDA(const Plato::OrdinalType aCellOrdinal, const Plato::OrdinalType aDofOrdinal, Plato::Scalar& tSumResult)
    {
        tSumResult += tResult(aCellOrdinal, aDofOrdinal);
    }, tSumResult );
    return tSumResult;
}
}

TEUCHOS_UNIT_TEST(NaturalBCsTests, UniformPressureConstructor)
{
    auto tInputParams = pressureBCParameters(
        "  <ParameterList  name='Natural Boundary Conditions'>\n"
        "    <ParameterList  name='Pressure Boundary Condition'>\n"
        "      <Parameter name='Type' type='string' value='Uniform pressure'/>\n"
        "      <Parameter name='Value' type='double' value='1'/>\n"
        "      <Parameter name='Sides' type='string' value='x+'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n");

    testBCDataConstruction<Plato::UniformNaturalBCData<3, Plato::BCDataType::kScalar>>(*tInputParams, out, success);
}

TEUCHOS_UNIT_TEST(NaturalBCsTests, UniformLoadConstructor)
{
    auto tInputParams = pressureBCParameters(
        "  <ParameterList  name='Natural Boundary Conditions'>\n"
        "    <ParameterList  name='Traction Boundary Condition'>\n"
        "      <Parameter name='Type' type='string' value='Uniform'/>\n"
        "      <Parameter name='Values' type='Array(double)' value='{1, 1, 1}'/>\n"
        "      <Parameter name='Sides' type='string' value='x+'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n");

    testBCDataConstruction<Plato::UniformNaturalBCData<3, Plato::BCDataType::kVector>>(*tInputParams, out, success);
}

TEUCHOS_UNIT_TEST(NaturalBCsTests, UniformLoadConstructorIndex)
{
    auto tInputParams = pressureBCParameters(
        "  <ParameterList  name='Natural Boundary Conditions'>\n"
        "    <ParameterList  name='Traction Boundary Condition'>\n"
        "      <Parameter name='Type' type='string' value='Uniform'/>\n"
        "      <Parameter name='Value' type='double' value='1'/>\n"
        "      <Parameter name='Index' type='int' value='2'/>\n"
        "      <Parameter name='Sides' type='string' value='x+'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n");

    testBCDataConstruction<Plato::UniformNaturalBCData<3, Plato::BCDataType::kVector>>(*tInputParams, out, success);
}

TEUCHOS_UNIT_TEST(NaturalBCsTests, TimeVaryingPressureConstructor)
{
    auto tInputParams = pressureBCParameters(
        "  <ParameterList  name='Natural Boundary Conditions'>\n"
        "    <ParameterList  name='Pressure Boundary Condition'>\n"
        "      <Parameter name='Type' type='string' value='Uniform pressure'/>\n"
        "      <Parameter name='Value' type='string' value='t * 42'/>\n"
        "      <Parameter name='Sides' type='string' value='x+'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n");

    testBCDataConstruction<Plato::TimeVaryingNaturalBCData<3, Plato::BCDataType::kScalar>>(*tInputParams, out, success);
}

TEUCHOS_UNIT_TEST(NaturalBCsTests, TimeVaryingLoadConstructor)
{
    auto tInputParams = pressureBCParameters(
        "  <ParameterList  name='Natural Boundary Conditions'>\n"
        "    <ParameterList  name='Traction Boundary Condition'>\n"
        "      <Parameter name='Type' type='string' value='Uniform'/>\n"
        "      <Parameter name='Values' type='Array(string)' value='{t * 42, t * 3.14159, t * 1.61803}'/>\n"
        "      <Parameter name='Sides' type='string' value='x+'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n");

    testBCDataConstruction<Plato::TimeVaryingNaturalBCData<3, Plato::BCDataType::kVector>>(*tInputParams, out, success);
}

TEUCHOS_UNIT_TEST(NaturalBCsTests, VaryingPressureConstructor)
{
    auto tInputParams = pressureBCParameters(
        "  <ParameterList  name='Natural Boundary Conditions'>\n"
        "    <ParameterList  name='Pressure Boundary Condition'>\n"
        "      <Parameter name='Type' type='string' value='Variable pressure'/>\n"
        "      <Parameter name='Variable' type='string' value='Pressure'/>\n"
        "      <Parameter name='Sides' type='string' value='x+'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n");
    using ExpectedType = Plato::SpatiallyVaryingNaturalBCData<3, Plato::BCDataType::kScalar>;
    testBCDataConstruction<ExpectedType>(*tInputParams, out, success);
}

TEUCHOS_UNIT_TEST(NaturalBCsTests, VaryingLoadConstructor)
{
    auto tInputParams = pressureBCParameters(
        "  <ParameterList  name='Natural Boundary Conditions'>\n"
        "    <ParameterList  name='Pressure Boundary Condition'>\n"
        "      <Parameter name='Type' type='string' value='Variable load'/>\n"
        "      <Parameter name='Variables' type='Array(string)' value='{pizza, tacos, french_fries}'/>\n"
        "      <Parameter name='Sides' type='string' value='x+'/>\n"
        "    </ParameterList>\n"
        "  </ParameterList>\n");
    using ExpectedType = Plato::SpatiallyVaryingNaturalBCData<3, Plato::BCDataType::kVector>;
    testBCDataConstruction<ExpectedType>(*tInputParams, out, success);
}

TEUCHOS_UNIT_TEST(NaturalBCsTests, BadInput)
{
    using ElementType = Plato::MechanicsElement<Plato::Tet4>;
    // Bad "Variable" name
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Pressure Boundary Condition'>\n"
            "      <Parameter name='Type' type='string' value='Variable pressure'/>\n"
            "      <Parameter name='Variabl' type='string' value='Pressure'/>\n"
            "      <Parameter name='Sides' type='string' value='x+'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        TEST_THROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")), std::runtime_error);
    }
    // Bad "Variable" type
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Pressure Boundary Condition'>\n"
            "      <Parameter name='Type' type='string' value='Variable pressure'/>\n"
            "      <Parameter name='Variable' type='int' value='1'/>\n"
            "      <Parameter name='Sides' type='string' value='x+'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        TEST_THROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")), std::runtime_error);
    }
    // Multiple valid types
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Pressure Boundary Condition'>\n"
            "      <Parameter name='Type' type='string' value='Uniform'/>\n"
            "      <Parameter name='Variable' type='string' value='burrito'/>\n"
            "      <Parameter name='Value' type='double' value='1'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        TEST_THROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")), std::runtime_error);
    }
    // Bad "Value" value 
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Pressure Boundary Condition'>\n"
            "      <Parameter name='Type' type='string' value='Uniform pressure'/>\n"
            "      <Parameter name='Value' type='double' value='hat'/>\n"
            "      <Parameter name='Sides' type='string' value='x+'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        // Ultimately, we want this to throw, but for strings not convertible to double, Teuchos will silently convert to 0. 
        // Test this for now in case it changes.
        TEST_NOTHROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")));
        // Test that a value of "hat" gives 0:
        const double tValue = tInputParams->sublist("Natural Boundary Conditions").sublist("Pressure Boundary Condition").get<double>("Value");
        TEST_EQUALITY(tValue, 0.0);
    }
    // Bad "Type" value
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Pressure Boundary Condition'>\n"
            "      <Parameter name='Type' type='string' value='pizza'/>\n"
            "      <Parameter name='Variable' type='int' value='1'/>\n"
            "      <Parameter name='Sides' type='string' value='x+'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        TEST_THROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")), std::runtime_error);
    }
    // Bad "Type" type
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Pressure Boundary Condition'>\n"
            "      <Parameter name='Type' type='double' value='3.14159'/>\n"
            "      <Parameter name='Variable' type='int' value='1'/>\n"
            "      <Parameter name='Sides' type='string' value='x+'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        TEST_THROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")), std::runtime_error);
    }
    // Bad "Sides" name
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Pressure Boundary Condition'>\n"
            "      <Parameter name='Type' type='string' value='Uniform pressure'/>\n"
            "      <Parameter name='Value' type='double' value='1'/>\n"
            "      <Parameter name='Side' type='string' value='x'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        TEST_THROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")), std::runtime_error);
    }
    // Bad "Sides" type 
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Pressure Boundary Condition'>\n"
            "      <Parameter name='Type' type='string' value='Uniform pressure'/>\n"
            "      <Parameter name='Value' type='double' value='1'/>\n"
            "      <Parameter name='Sides' type='double' value='42.0'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        TEST_THROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")), std::runtime_error);
    }
    // Bad number of components
    {
        auto tInputParams = pressureBCParameters(
            "  <ParameterList  name='Natural Boundary Conditions'>\n"
            "    <ParameterList  name='Load Boundary Condition'>\n"
            "      <Parameter name='Type' type='string' value='Uniform'/>\n"
            "      <Parameter name='Values' type='Array(double)' value='{1.0, 0.0}'/>\n"
            "      <Parameter name='Sides' type='string' value='x+'/>\n"
            "    </ParameterList>\n"
            "  </ParameterList>\n");
        TEST_THROW(Plato::NaturalBCs<ElementType>(tInputParams->sublist("Natural Boundary Conditions")), std::runtime_error);
    }
    // Bad nodal variable name
    {
        auto tInputParams = pressureBCParameters(
            "<ParameterList  name='Natural Boundary Conditions'>\n"
            "  <ParameterList  name='Load Boundary Condition'>\n"
            "    <Parameter name='Type' type='string' value='Variable pressure'/>\n"
            "    <Parameter name='Variable' type='string' value='surface_pressure_nope'/>\n"
            "    <Parameter name='Sides' type='string' value='pressure_sideset'/>\n"
            "  </ParameterList>\n"
            "</ParameterList>\n");

        TEST_THROW(surfaceIntegralSum(*tInputParams), std::runtime_error);
    }
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, SurfaceIntegralSpatiallyVarying)
{
    auto tInputParams = pressureBCParameters(
        "<ParameterList  name='Natural Boundary Conditions'>\n"
        "  <ParameterList  name='Pressure Boundary Condition'>\n"
        "    <Parameter name='Type' type='string' value='Variable pressure'/>\n"
        "    <Parameter name='Variable' type='string' value='surface_pressure'/>\n"
        "    <Parameter name='Sides' type='string' value='pressure_sideset'/>\n"
        "  </ParameterList>\n"
        "</ParameterList>\n");

    const Plato::Scalar tSumResult = surfaceIntegralSum(*tInputParams);

    // The boundary condition on the mesh is 10 + y + z, 
    // integrating this analytically over the face:
    // \[ \int_{-5}^5 \int_{-5}^5 10 + y + z \,dy \,dz \]
    // gives the answer of 1000
    
    // test to see if the sum of the entries in tResult is
    // equal to the analytic integral
    TEST_FLOATING_EQUALITY(tSumResult, 1000, 1e-15);
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, SurfaceIntegralSpatiallyVaryingLoad)
{
    auto tInputParams = pressureBCParameters(
        "<ParameterList  name='Natural Boundary Conditions'>\n"
        "  <ParameterList  name='Load Boundary Condition'>\n"
        "    <Parameter name='Type' type='string' value='Variable load'/>\n"
        "    <Parameter name='Variables' type='Array(string)' value='{surface_pressure, surface_pressure, surface_pressure}'/>\n"
        "    <Parameter name='Sides' type='string' value='pressure_sideset'/>\n"
        "  </ParameterList>\n"
        "</ParameterList>\n");

    const Plato::Scalar tSumResult = surfaceIntegralSum(*tInputParams);

    // The boundary condition on the mesh is 10 + y + z, 
    // integrating this analytically over the face:
    // \[ \int_{-5}^5 \int_{-5}^5 10 + y + z \,dy \,dz \]
    // gives the answer of 1000
    // For a load, this is done 3 times, so results is 3000
    TEST_FLOATING_EQUALITY(tSumResult, 3000, 1e-15);
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, SurfaceIntegralSpatiallyVaryingLoadWithZero)
{
    auto tInputParams = pressureBCParameters(
        "<ParameterList  name='Natural Boundary Conditions'>\n"
        "  <ParameterList  name='Load Boundary Condition'>\n"
        "    <Parameter name='Type' type='string' value='Variable load'/>\n"
        "    <Parameter name='Variables' type='Array(string)' value='{0, surface_pressure, 0}'/>\n"
        "    <Parameter name='Sides' type='string' value='pressure_sideset'/>\n"
        "  </ParameterList>\n"
        "</ParameterList>\n");

    const Plato::Scalar tSumResult = surfaceIntegralSum(*tInputParams);

    // The boundary condition on the mesh is 10 + y + z, 
    // integrating this analytically over the face:
    // \[ \int_{-5}^5 \int_{-5}^5 10 + y + z \,dy \,dz \]
    // gives the answer of 1000
    // In this case, only 1 component is set so answer is still 1000
    TEST_FLOATING_EQUALITY(tSumResult, 1000, 1e-15);
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, SurfaceIntegralUniform)
{
    auto tInputParams = pressureBCParameters(
        "<ParameterList  name='Natural Boundary Conditions'>\n"
        "  <ParameterList  name='Pressure Boundary Condition'>\n"
        "    <Parameter name='Type' type='string' value='Uniform pressure'/>\n"
        "    <Parameter name='Value' type='string' value='1.0'/>\n"
        "    <Parameter name='Sides' type='string' value='pressure_sideset'/>\n"
        "  </ParameterList>\n"
        "</ParameterList>\n");

    const Plato::Scalar tSumResult = surfaceIntegralSum(*tInputParams);

    constexpr Plato::Scalar kUniformPressure = 1.0;
    constexpr Plato::Scalar kSideLength = 1.0;
    constexpr Plato::OrdinalType kNumFaces = 100;
    constexpr Plato::Scalar kExpectedResult = kUniformPressure * kSideLength * kSideLength * kNumFaces;
    TEST_FLOATING_EQUALITY(tSumResult, kExpectedResult, 1e-15);
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, SurfaceIntegralUniformLoad)
{
    auto tInputParams = pressureBCParameters(
        "<ParameterList  name='Natural Boundary Conditions'>\n"
        "  <ParameterList  name='Load Boundary Condition'>\n"
        "    <Parameter name='Type' type='string' value='Uniform'/>\n"
        "    <Parameter name='Values' type='Array(double)' value='{1.0, 0.0, 0.0}'/>\n"
        "    <Parameter name='Sides' type='string' value='pressure_sideset'/>\n"
        "  </ParameterList>\n"
        "</ParameterList>\n");

    const Plato::Scalar tSumResult = surfaceIntegralSum(*tInputParams);

    constexpr Plato::Scalar kUniformPressure = 1.0;
    constexpr Plato::Scalar kSideLength = 1.0;
    constexpr Plato::OrdinalType kNumFaces = 100;
    constexpr Plato::Scalar kExpectedResult = kUniformPressure * kSideLength * kSideLength * kNumFaces;
    TEST_FLOATING_EQUALITY(tSumResult, kExpectedResult, 1e-15);
}
