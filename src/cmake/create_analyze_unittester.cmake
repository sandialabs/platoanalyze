# create_analyze_unittester
#   TEST_LIB: The name of the library to create a unit tester for. The test exe name will be
#    the name of the library with `_UnitTests` appended.
#   DIRECTORIES_OR_FILES: A list of directories or files containing the test source. All cpp
#    files in a given directory will be globbed.
function( create_analyze_unittester TEST_LIB DIRECTORIES_OR_FILES)

    unset(TEST_SRCS)
    foreach(DIRECTORY_OR_FILE ${DIRECTORIES_OR_FILES})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY_OR_FILE})
            file(GLOB EXT_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${DIRECTORY_OR_FILE}/*.cpp)
            list(APPEND TEST_SRCS ${EXT_SRCS})
        elseif(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY_OR_FILE})
            list(APPEND TEST_SRCS ${DIRECTORY_OR_FILE})
        else()
            message(FATAL_ERROR "File or directory not found when creating unit tester ${TEST_LIB}: ${DIRECTORY_OR_FILE}")
        endif()
    endforeach(DIRECTORY_OR_FILE)

    list(APPEND TEST_SRCS ${UNIT_TEST_MAIN_CPP})

    set(TEST_EXE "${TEST_LIB}_UnitTests")
    add_executable(${TEST_EXE} ${TEST_SRCS})

    target_link_libraries(${TEST_EXE} PUBLIC PlatoEngine::PlatoTestUtilitiesInterface PRIVATE Analyze_UnitTestUtils Trilinos::all_selected_libs)
    target_include_directories(${TEST_EXE} PRIVATE ${PROJECT_SOURCE_DIR}/unit_tests/util )

    add_test(NAME ${TEST_EXE} COMMAND ${TEST_EXE})

endfunction(create_analyze_unittester)
