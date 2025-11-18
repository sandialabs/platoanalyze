include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

set(LINK_LIBS AnalyzeDomain 
              AnalyzeParsing)
create_plato_analyze_library(AnalyzeMaterial "." "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST)
    set(LINK_LIBS AnalyzeTestUtilities)  
    set(FOLDERS unittest)
    create_plato_analyze_teuchos_unittester(AnalyzeMaterial "${FOLDERS}" "${LINK_LIBS}")

endif()
