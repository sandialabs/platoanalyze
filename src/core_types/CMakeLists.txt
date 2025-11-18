include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

set(LINK_LIBS Sacado::sacado 
              Kokkos::kokkos)
create_plato_analyze_header_library(AnalyzeCoreTypes "." "${LINK_LIBS}")
