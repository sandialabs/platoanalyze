include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

add_subdirectory(./BamG)

set(LINK_LIBS AnalyzeCoreTypes 
              AnalyzeLinearAlgebra 
              AnalyzeDomain
              AnalyzeMesh
              AnalyzeUtilities
              AnalyzeBamG
              STKIO::stk_io
              )
create_plato_analyze_library(AnalyzeTestUtilities "." "${LINK_LIBS}")
