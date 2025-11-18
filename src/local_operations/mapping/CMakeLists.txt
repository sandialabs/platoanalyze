include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

set(LINK_LIBS AnalyzeLinearAlgebra )
create_plato_analyze_header_library(AnalyzeMappingOperations "." "${LINK_LIBS}")
