# Test utility helper library

set( TEST_UTIL_SRC 
     PlatoTestHelpers.cpp 
     PlatoMathTestHelpers.cpp )

add_library( Analyze_UnitTestUtils ${TEST_UTIL_SRC} )

target_include_directories( Analyze_UnitTestUtils PRIVATE . )
target_include_directories( Analyze_UnitTestUtils PRIVATE ${PROJECT_SOURCE_DIR}/src )
target_include_directories( Analyze_UnitTestUtils PRIVATE ${PLATOENGINE_PREFIX}/include )
target_include_directories( Analyze_UnitTestUtils PRIVATE ${PROJECT_SOURCE_DIR}/unit_tests/BamG )

set( TEST_UTIL_LINK_LIBS
     analyzelib
     BamGlib
     ${PLATO_LIBS}
     Trilinos::all_selected_libs )

target_link_libraries( Analyze_UnitTestUtils PUBLIC ${TEST_UTIL_LINK_LIBS} )