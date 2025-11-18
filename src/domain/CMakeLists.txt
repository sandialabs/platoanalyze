include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

set(LINK_LIBS AnalyzeCoreTypes)
set(FOLDERS "."
            contact)

if(PLATOANALYZE_ENABLE_MESHMAP)
    list(APPEND FOLDERS "contact/multipoint_constraint")
endif()
if (PLATOANALYZE_ENABLE_MESHMAP)
  list(APPEND LINK_LIBS ArborX::ArborX)
endif()


create_plato_analyze_library(AnalyzeDomain "${FOLDERS}" "${LINK_LIBS}")
if( PLATOANALYZE_UNIT_TEST)
    set(LINK_LIBS AnalyzeTestUtilities)  
    set(FOLDERS unittest)

    create_plato_analyze_teuchos_unittester(AnalyzeDomain "${FOLDERS}" "${LINK_LIBS}")

endif()
