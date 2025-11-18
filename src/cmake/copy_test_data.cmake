# copy_test_data
#   Copies the files COPY_FILES from ${ANALYZE_TEST_DATA_DIR} to the build and install paths.
macro( copy_test_data COPY_FILES )
    foreach( CP_FILE ${COPY_FILES} )
        file(COPY ${ANALYZE_TEST_DATA_DIR}/${CP_FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
        file(COPY ${ANALYZE_TEST_DATA_DIR}/${CP_FILE} DESTINATION ${CMAKE_INSTALL_PREFIX})
    endforeach(CP_FILE)
endmacro(copy_test_data)
