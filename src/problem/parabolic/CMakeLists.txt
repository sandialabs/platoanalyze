
include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS ".")
set(LINK_LIBS AnalyzeProblem AnalyzeGeometric )
create_plato_analyze_library(AnalyzeParabolic "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST )

  set(FOLDERS "test_utilities") 
  set(LINK_LIBS TeuchosCore::teuchoscore
                AnalyzeTestUtilities)
  create_plato_analyze_library(AnalyzeParabolicTestUtilities "${FOLDERS}" "${LINK_LIBS}")

  set(FOLDERS "unittest") 
  set(LINK_LIBS AnalyzeParabolicTestUtilities
                AnalyzeUtilities) 
  create_plato_analyze_teuchos_unittester(AnalyzeParabolic "${FOLDERS}" "${LINK_LIBS}")
endif()
