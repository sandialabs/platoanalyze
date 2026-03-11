
include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS "library")
set(LINK_LIBS AnalyzeProblem 
              AnalyzeGeometric)


if(ELLIPTIC)
  list(APPEND LINK_LIBS AnalyzeElliptic)
endif()
if(PARABOLIC)
  list(APPEND LINK_LIBS AnalyzeParabolic)
endif()

if(HYPERBOLIC)
  list(APPEND LINK_LIBS AnalyzeHyperbolic)
endif()

list(APPEND LINK_LIBS AnalyzeHelmholtz)


create_plato_analyze_header_library(AnalyzeLib "${FOLDERS}" "${LINK_LIBS}")
SET_TARGET_PROPERTIES(AnalyzeLib PROPERTIES OUTPUT_NAME analyze)
if( CMAKE_INSTALL_PREFIX )
  install( TARGETS AnalyzeLib DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
endif()
