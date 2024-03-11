set(Test_SOURCES Test_FunctionalInterfaceUtilities.cpp
                 Test_SolutionCache.cpp
                 UnitMain.cpp)

set(Test_NAME "AnalyzeFunctional_UnitTests")
add_executable(${Test_NAME} ${Test_SOURCES})

target_link_libraries(${Test_NAME} PRIVATE Analyze_UnitTestUtils AnalyzeFunctionalInterface Trilinos::all_selected_libs )
target_include_directories(${Test_NAME} PRIVATE .)
target_include_directories(${Test_NAME} PRIVATE "${PLATOENGINE_PREFIX}/include")
target_include_directories(${Test_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/unit_tests/util )

add_test(NAME ${Test_NAME} COMMAND ${Test_NAME})
