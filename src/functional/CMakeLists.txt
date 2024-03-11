set(SRCS AnalyzeCriterionInterface.cpp
         FunctionalInterface.cpp
         FunctionalInterfaceUtilities.cpp
         SolutionCache.cpp
         HelmholtzFilterInterface.cpp)

set(HDRS AnalyzeCriterionInterface.hpp
         FunctionalInterface.hpp
         FunctionalInterfaceUtilities.hpp
         SolutionCache.hpp
         HelmholtzFilterInterface.hpp)

set(LIB_NAME AnalyzeFunctionalInterface)
add_library(${LIB_NAME} SHARED ${SRCS} ${HDRS} )
target_include_directories(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)
target_include_directories(${LIB_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src/alg )
target_include_directories(analyzelib PUBLIC "${PLATOENGINE_PREFIX}/include")
target_link_libraries(${LIB_NAME} PUBLIC analyzelib)

install( TARGETS ${LIB_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if( PLATOANALYZE_UNIT_TEST )
  add_subdirectory(./unittest)
endif()
