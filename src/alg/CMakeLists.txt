add_subdirectory(./nonlinear_solvers)

if( PLATOANALYZE_UNIT_TEST )
  include(${CMAKE_UTIL_DIR}/create_analyze_unittester.cmake)

  set(SOLVER_TEST_FILES unittest/CrsMatrixUtilsTests.cpp)

  if(PLATOANALYZE_ENABLE_TACHO)
    list(APPEND SOLVER_TEST_FILES unittest/TachoSolverTests.cpp)
  endif()

  if(PLATOANALYZE_ENABLE_UMFPACK)
    list(APPEND SOLVER_TEST_FILES unittest/SuiteSparseUtilsTests.cpp
                                  unittest/SuiteSparseSolverFactoryTests.cpp
                                  unittest/CHOLMODLinearSolverTests.cpp
                                  unittest/UMFPACKLinearSolverTests.cpp)
  endif()

  create_analyze_unittester(AnalyzeAlgorithmsLinearSolvers "${SOLVER_TEST_FILES}")
endif()
