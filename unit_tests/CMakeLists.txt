configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/amgx.json
                 ${CMAKE_CURRENT_BINARY_DIR}/amgx.json COPYONLY)
configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/points.xyz
                 ${CMAKE_CURRENT_BINARY_DIR}/points.xyz COPYONLY)
configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/data/brick_with_data.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/brick_with_data.exo COPYONLY)
configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/data/nodal_surface_pressure_field.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/nodal_surface_pressure_field.exo COPYONLY)
configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/data/two_block_contact.exo
                 ${CMAKE_CURRENT_BINARY_DIR}/two_block_contact.exo COPYONLY)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/BamG)

include (${CMAKE_CURRENT_SOURCE_DIR}/add_test.cmake)

# Build test helper library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/util)

## if SKIP_MATH is provided as the optional third argument below, unit tests
## will be compiled without checking for math exceptions.

Plato_add_test(PlatoMathHelpers PlatoMathHelpersTest.cpp)

#TODO: update StructuralDynamics to new element
#Plato_add_test(PlatoFunctors PlatoFunctorsTest.cpp)

Plato_add_test(ExpressionEvaluator ExpressionEvaluatorTest.cpp)
Plato_add_test(MassProperties MassPropertiesTests.cpp)
Plato_add_test(GeometryMisfit GeometryMisfitTests.cpp)
Plato_add_test(LinearMaterialModel LinearMaterialModelTests.cpp)
Plato_add_test(PlatoMaterialModel PlatoMaterialModelTest.cpp)
Plato_add_test(MaterialBasis MaterialBasisTest.cpp)
Plato_add_test(ReadData ReadDataTest.cpp)
Plato_add_test(PlatoSpatialModel PlatoSpatialModelTests.cpp)
Plato_add_test(Element ElementTests.cpp)
Plato_add_test(PlatoMathTypes PlatoMathTypesTests.cpp)
Plato_add_test(TensorNorm TensorNormTests.cpp)
Plato_add_test(CrsMatrixUtils CrsMatrixUtilsTests.cpp)
Plato_add_test(Assembly AssemblyTests.cpp)
Plato_add_test(NaturalBCsTests NaturalBCsTests.cpp)
Plato_add_test(NaturalBCDataTests NaturalBCDataTests.cpp)

if(ELLIPTIC)
Plato_add_test(MultipointConstraint PlatoMultipointConstraintTests.cpp)
Plato_add_test(SolverInterface PlatoSolverInterfaceTests.cpp)
Plato_add_test(SolutionFunction SolutionFunctionTests.cpp)
Plato_add_test(VolumeAverageCriterion VolumeAverageCriterionTests.cpp SKIP_MATH)
Plato_add_test(AugLagStress PlatoAugLagStressTest.cpp)
Plato_add_test(Elastic_Tet10 ElasticTests_Tet10.cpp)
Plato_add_test(Thermostatic ThermostaticTests.cpp)
Plato_add_test(Elastostatic ElastostaticTests.cpp)
Plato_add_test(Electroelastic ElectroelasticTests.cpp)
Plato_add_test(Thermoelastic ThermoelasticTests.cpp)
endif()

if(HATCHING)
  Plato_add_test(EllipticHatching EllipticHatchingTests.cpp)
endif()

if(STABILIZED)
  Plato_add_test(TwoField TwoFieldTests.cpp)
  Plato_add_test(StabilizedMechanics StabilizedMechanicsTests.cpp)
endif()

if(PLASTICITY)
  Plato_add_test(Plasticity PlasticityTests.cpp)
  Plato_add_test(ElastoPlasticity ElastoPlasticityTests.cpp SKIP_MATH)
  Plato_add_test(Thermoplasticity ThermoplasticityTests.cpp SKIP_MATH)
endif()

if(PARABOLIC)
  Plato_add_test(TransientThermomech TransientThermomechTests.cpp)
  Plato_add_test(HeatEquation HeatEquationTests.cpp)
endif()

if(HYPERBOLIC)
  Plato_add_test(TransientDynamics TransientDynamicsTests.cpp)
  if(FLUIDS)
    Plato_add_test(ComputationalFluidDynamics ComputationalFluidDynamicsTests.cpp SKIP_MATH)
  endif()
  if(MICROMORPHIC)
    Plato_add_test(RelaxedMicromorphic RelaxedMicromorphicTests.cpp)
  endif()
endif()

if(HELMHOLTZ)
  Plato_add_test(HelmholtzFilter PlatoHelmholtzFilterTests.cpp)
  Plato_add_test(HelmholtzMatrix PlatoHelmholtzMatrixTests.cpp)
  set_tests_properties(runHelmholtzMatrixUnitTests PROPERTIES WILL_FAIL true) # until quadrature rules for tet4, tri3 are fixed
endif()

if(PLATOANALYZE_ENABLE_TACHO)
  Plato_add_test(TachoSolver TachoSolverTests.cpp)
endif()

if(PLATOANALYZE_ENABLE_UMFPACK)
  Plato_add_test(UMFPACKSolver UMFPACKSolverTests.cpp)
endif()

if(PLATOANALYZE_ENABLE_MPMD)
  set(AnalyzeApp_SOURCES
    AnalyzeAppIntxTests.cpp
    PlatoUnitTestMain.cpp
  )
  add_executable(AnalyzeAppIntxTests
    ${AnalyzeApp_SOURCES}
  )
  target_link_libraries(AnalyzeAppIntxTests
    Analyze_App
    analyzelib
    ${PLATO_LIBS}
    Trilinos::all_selected_libs
    )

  target_include_directories(AnalyzeAppIntxTests PRIVATE "${PLATOENGINE_PREFIX}/include")
  target_include_directories(AnalyzeAppIntxTests PRIVATE "${CMAKE_SOURCE_DIR}/src")
  target_include_directories(AnalyzeAppIntxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BamG")
  target_compile_definitions(AnalyzeAppIntxTests PRIVATE BOOST_NO_AUTO_PTR=1) # Suppress warnings on nvcc build

  if (PLATOANALYZE_ENABLE_ENGINEMESH)
    set(EngineMesh_SOURCES
      EngineMeshIntxTests.cpp
      PlatoUnitTestMain.cpp
    )
    add_executable(EngineMeshIntxTests
      ${EngineMesh_SOURCES}
    )
    target_link_libraries(EngineMeshIntxTests
      analyzelib
      BamGlib
      ${PLATO_LIBS}
      Trilinos::all_selected_libs
      Analyze_UnitTestUtils
      )

    target_include_directories(EngineMeshIntxTests PRIVATE "${PLATOENGINE_PREFIX}/include")
    target_include_directories(EngineMeshIntxTests PRIVATE "${CMAKE_SOURCE_DIR}/src")
    target_include_directories(EngineMeshIntxTests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/BamG")
  endif()

  if(PLATOANALYZE_ENABLE_MESHMAP)
    Plato_add_test(PlatoMeshMap PlatoMeshMapTests.cpp)
    Plato_add_test(Contact ContactTests.cpp)
  endif()

endif()
