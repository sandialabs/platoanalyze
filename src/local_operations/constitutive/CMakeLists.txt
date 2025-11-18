include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)

set(LINK_LIBS AnalyzeCoreTypes 
              AnalyzeLinearAlgebra
              AnalyzeMaterial )
create_plato_analyze_header_library(AnalyzeConstitutiveOperations "." "${LINK_LIBS}")
