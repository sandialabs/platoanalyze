include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(LINK_LIBS AnalyzeCoreTypes  
              Kokkos::kokkos 
              TeuchosCore::teuchoscore)
create_plato_analyze_library(AnalyzeLinearAlgebra "." "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST)
    set(LINK_LIBS AnalyzeTestUtilities)  
    set(FOLDERS unittest)

    create_plato_analyze_teuchos_unittester(AnalyzeLinearAlgebra "${FOLDERS}" "${LINK_LIBS}")

endif()
