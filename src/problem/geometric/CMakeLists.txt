include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)
include(${CMAKE_UTIL_DIR}/copy_test_data.cmake)

set(COPY_FILES  points.xyz)
copy_test_data("${COPY_FILES}")

set(FOLDERS ".")

set(LINK_LIBS AnalyzeProblem)
create_plato_analyze_library(AnalyzeGeometric "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "unittest") 
  
  set(LINK_LIBS AnalyzeTestUtilities AnalyzeUtilities) 

  create_plato_analyze_teuchos_unittester(AnalyzeGeometric "${FOLDERS}" "${LINK_LIBS}")
endif()
