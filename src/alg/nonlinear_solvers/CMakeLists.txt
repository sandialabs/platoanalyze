if( PLATOANALYZE_UNIT_TEST )
  include(${CMAKE_UTIL_DIR}/create_analyze_unittester.cmake)
  set(DIRECTORIES unittest)

  create_analyze_unittester(AnalyzeAlgorithmsNonlinearSolvers ${DIRECTORIES})
endif()