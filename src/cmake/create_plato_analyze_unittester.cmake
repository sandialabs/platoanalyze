include(${CMAKE_UTIL_DIR}/add_to_srcs_and_hdrs.cmake)

# create_plato_analyze_unittester
#   TEST_LIB: The name of the library to create a unit tester for. The test exe name will be
#    the name of the library with `_UnitTester` appended.
#   DIRECTORIES: A list of directories containing the test source files.
#   Additional targets to link with may be passed as extra arguments.
function( create_plato_analyze_unittester TEST_LIB DIRECTORIES )

    set(EXTRA_LIBS ${ARGN})
    set(TEST_EXE "${TEST_LIB}_UnitTester")
    set(TARGET_LINK_LIST "${TEST_LIB}")
    create_plato_analyze_unittester_impl( ${TEST_EXE} "${DIRECTORIES}" ${TEST_UNIT_MAIN_INCL} "${TARGET_LINK_LIST};${EXTRA_LIBS}" GTest::GTest "--gtest_output=xml:${TEST_EXE}.xml")

endfunction(create_plato_analyze_unittester)


# create_plato_analyze_teuchos_unittester
#   TEST_LIB: The name of the library to create a unit tester for. The test exe name will be
#    the name of the library with `_UnitTester` appended.
#   DIRECTORIES: A list of directories containing the test source files.
#   Additional targets to link with may be passed as extra arguments.
function( create_plato_analyze_teuchos_unittester TEST_LIB DIRECTORIES )

    set(EXTRA_LIBS ${ARGN})
    set(TEST_EXE "${TEST_LIB}_UnitTester")
    set(TARGET_LINK_LIST "${TEST_LIB}")
    create_plato_analyze_unittester_impl( ${TEST_EXE} "${DIRECTORIES}" ${TEST_TEUCHOS_UNIT_MAIN_INCL} "${TARGET_LINK_LIST};${EXTRA_LIBS}" TeuchosCore::teuchoscore "")

endfunction(create_plato_analyze_teuchos_unittester)

function( create_plato_analyze_unittester_impl TEST_EXE DIRECTORIES TEST_MAIN_CPP TARGET_LINK_LIST GTEST_OR_TEUCHOS_LIB GTEST_OUTPUT)

    unset(TEST_SRCS)
    unset(TEST_HDRS)

    foreach( curDir ${DIRECTORIES} )
        add_to_srcs_and_hdrs(${curDir} TEST_SRCS TEST_HDRS)
    endforeach(curDir)
    
    list(APPEND TEST_SRCS ${TEST_MAIN_CPP})

    add_executable(${TEST_EXE} ${TEST_SRCS} ${TEST_HDRS})
    target_compile_options(${TEST_EXE} PRIVATE "-fPIC")

    target_link_libraries( ${TEST_EXE} PRIVATE ${GTEST_OR_TEUCHOS_LIB}  ${TARGET_LINK_LIST})
    add_test(NAME ${TEST_EXE} COMMAND ${TEST_EXE} ${GTEST_OUTPUT})
    set_property(TEST ${TEST_EXE} PROPERTY LABELS "small")

    install( TARGETS ${TEST_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )

endfunction(create_plato_analyze_unittester_impl)
