include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

set(LINK_LIBS AnalyzeCoreTypes)
create_plato_analyze_header_library(AnalyzeSimplex "." "${LINK_LIBS}")
