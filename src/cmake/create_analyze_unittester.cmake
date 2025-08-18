# create_analyze_unittester
#   TEST_LIB: The name of the library to create a unit tester for. The test exe name will be
#    the name of the library with `_UnitTests` appended.
#   DIRECTORIES: A list of directories containing the test source files.
function( create_analyze_unittester TEST_LIB DIRECTORIES)

    unset(TEST_SRCS)
    foreach( DIRECTORY ${DIRECTORIES} )
        file(GLOB EXT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${DIRECTORY}/*.cpp)
        list(APPEND TEST_SRCS ${EXT_SRCS})
    endforeach(DIRECTORY)
    list(APPEND TEST_SRCS ${UNIT_TEST_MAIN_CPP})

    set(TEST_EXE "${TEST_LIB}_UnitTests")
    add_executable(${TEST_EXE} ${TEST_SRCS})

    target_link_libraries(${TEST_EXE} PUBLIC PlatoEngine::PlatoTestUtilitiesInterface PRIVATE Analyze_UnitTestUtils Trilinos::all_selected_libs)
    target_include_directories(${TEST_EXE} PRIVATE ${PROJECT_SOURCE_DIR}/unit_tests/util )

    add_test(NAME ${TEST_EXE} COMMAND ${TEST_EXE})

endfunction(create_analyze_unittester)
