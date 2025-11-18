include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(LINK_LIBS AnalyzeCoreTypes TeuchosCore::teuchoscore)
create_plato_analyze_library(AnalyzeParsing "." "${LINK_LIBS}")

if(PLATOANALYZE_UNIT_TEST )
  set(LINK_LIBS AnalyzeTestUtilities)  
  set(FOLDERS unittest)
  create_plato_analyze_teuchos_unittester(AnalyzeParsing "${FOLDERS}" "${LINK_LIBS}")
endif()
