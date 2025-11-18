include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/copy_test_data.cmake)

set(COPY_FILES  brick_with_data.exo
                nodal_surface_pressure_field.exo)
copy_test_data("${COPY_FILES}")

set(LINK_LIBS AnalyzeDomain
              TeuchosCore::teuchoscore)
create_plato_analyze_library(AnalyzeBoundaryConditions "." "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST)
    set(LINK_LIBS AnalyzeTestUtilities)  
    set(FOLDERS unittest)
    create_plato_analyze_teuchos_unittester(AnalyzeBoundaryConditions "${FOLDERS}" "${LINK_LIBS}")

endif()
