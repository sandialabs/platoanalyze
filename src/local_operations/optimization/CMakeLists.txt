include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

set(LINK_LIBS AnalyzeCoreTypes 
              AnalyzeDomain)
create_plato_analyze_header_library(AnalyzeOptimizationOperations "." "${LINK_LIBS}")
