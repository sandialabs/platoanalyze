include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

set(LINK_LIBS AnalyzeCoreTypes
              ArborX::ArborX)
set(FOLDERS "."
            contact
            contact/multipoint_constraint)

create_plato_analyze_library(AnalyzeDomain "${FOLDERS}" "${LINK_LIBS}")
if( PLATOANALYZE_UNIT_TEST)
    set(LINK_LIBS AnalyzeTestUtilities)  
    set(FOLDERS unittest)

    create_plato_analyze_teuchos_unittester(AnalyzeDomain "${FOLDERS}" "${LINK_LIBS}")

endif()
