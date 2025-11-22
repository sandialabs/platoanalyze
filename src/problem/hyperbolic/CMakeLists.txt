include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS ".")
set(LINK_LIBS AnalyzeProblem AnalyzeMesh )
create_plato_analyze_library(AnalyzeHyperbolic "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "unittest") 

  set(LINK_LIBS AnalyzeTestUtilities 
                AnalyzeUtilities 
                AnalyzeElliptic 
                AnalyzeParabolic 
                AnalyzeHelmholtz) 
  create_plato_analyze_teuchos_unittester(AnalyzeHyperbolic "${FOLDERS}" "${LINK_LIBS}")
endif()
