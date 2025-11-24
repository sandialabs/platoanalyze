function(set_openmp_test_properties TEST_NAME)

set( NUM_THREADS "1" )
if( PLATOANALYZE_ENABLE_OPENMP )
    set( NUM_THREADS "8" )
endif()

set_tests_properties( ${TEST_NAME} PROPERTIES PROCESSORS ${NUM_THREADS} ENVIRONMENT "OMP_NUM_THREADS=${NUM_THREADS};OMP_PROC_BIND=false;OMP_PLACES=threads")

endfunction(set_openmp_test_properties)
