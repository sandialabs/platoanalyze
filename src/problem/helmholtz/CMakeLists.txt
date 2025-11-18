include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS ".")
set(LINK_LIBS AnalyzeProblem )
create_plato_analyze_library(AnalyzeHelmholtz "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "unittest") 
  set(LINK_LIBS AnalyzeTestUtilities AnalyzeUtilities) 
  create_plato_analyze_teuchos_unittester(AnalyzeHelmholtz "${FOLDERS}" "${LINK_LIBS}")
endif()
