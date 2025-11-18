include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS ".")
if(FLUIDS)
  list(APPEND FOLDERS fluids)
endif()
if(MICROMORPHIC)
  list(APPEND FOLDERS micromorphic)
endif()

set(LINK_LIBS AnalyzeProblem AnalyzeMesh )
create_plato_analyze_library(AnalyzeHyperbolic "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "unittest") 
  if(FLUIDS)
    list(APPEND FOLDERS fluids/unittest)
  endif()
  if(MICROMORPHIC)
    list(APPEND FOLDERS micromorphic/unittest)
  endif()
  set(LINK_LIBS AnalyzeTestUtilities 
                AnalyzeUtilities 
                AnalyzeElliptic 
                AnalyzeParabolic 
                AnalyzeHelmholtz) 
  create_plato_analyze_teuchos_unittester(AnalyzeHyperbolic "${FOLDERS}" "${LINK_LIBS}")
endif()
