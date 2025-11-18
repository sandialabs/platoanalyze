include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS ".")
set(LINK_LIBS AnalyzeLib 
              PlatoEngine::PlatoCriteriaInterface 
              PlatoEngine::PlatoFilterInterface 
              PlatoEngine::PlatoLinearAlgebraInterface)
create_plato_analyze_library(AnalyzeFunctionalInterface "${FOLDERS}" "${LINK_LIBS}")

install( TARGETS AnalyzeFunctionalInterface DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(EXISTS "${PLATOENGINE_PREFIX}")
  message(STATUS "Installing platoanalyze to platoengine plugins directory at ${PLATOENGINE_PREFIX}")
  install(TARGETS AnalyzeFunctionalInterface DESTINATION ${PLATOENGINE_PREFIX}/plugins)
  install(FILES "platoanalyze.config" DESTINATION ${PLATOENGINE_PREFIX}/plugins)
endif()

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "unittest") 
  set(LINK_LIBS AnalyzeTestUtilities ) 
  create_plato_analyze_teuchos_unittester(AnalyzeFunctionalInterface "${FOLDERS}" "${LINK_LIBS}")
endif()
