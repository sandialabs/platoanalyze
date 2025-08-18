set(SRCS AnalyzeCriterionInterface.cpp
         FunctionalInterface.cpp
         FunctionalInterfaceUtilities.cpp
         HelmholtzFilterInterface.cpp
         InputValidation.cpp
         ParameterListUtilities.cpp)

set(HDRS AnalyzeCriterionInterface.hpp
         FunctionalInterface.hpp
         FunctionalInterfaceUtilities.hpp
         HelmholtzFilterInterface.hpp
         InputValidation.hpp
         ParameterListUtilities.hpp)

set(LIB_NAME AnalyzeFunctionalInterface)
add_library(${LIB_NAME} SHARED ${SRCS} ${HDRS} )
target_include_directories(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/alg )
target_link_libraries(${LIB_NAME} PUBLIC analyzelib PlatoEngine::PlatoCriteriaInterface PlatoEngine::PlatoFilterInterface PlatoEngine::PlatoLinearAlgebraInterface)

install( TARGETS ${LIB_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(EXISTS "${PLATOENGINE_PREFIX}")
  message(STATUS "Installing platoanalyze to platoengine plugins directory at ${PLATOENGINE_PREFIX}")
  install(TARGETS ${LIB_NAME} DESTINATION ${PLATOENGINE_PREFIX}/plugins)
  install(FILES "platoanalyze.config" DESTINATION ${PLATOENGINE_PREFIX}/plugins)
endif()

if( PLATOANALYZE_UNIT_TEST )
  add_subdirectory(./unittest)
endif()
