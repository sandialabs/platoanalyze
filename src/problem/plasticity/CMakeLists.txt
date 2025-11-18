include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS ".")
set(LINK_LIBS AnalyzeProblem )
create_plato_analyze_library(AnalyzePlasticity "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "unittest") 
  set(LINK_LIBS AnalyzeTestUtilities ) 
  create_plato_analyze_teuchos_unittester(AnalyzePlasticity "${FOLDERS}" "${LINK_LIBS}")
endif()
