include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)
include(${CMAKE_UTIL_DIR}/copy_test_data.cmake)

set(COPY_FILES  two_block_contact.exo )                
copy_test_data("${COPY_FILES}")

set(FOLDERS "."
            finite_deformation_mechanics)
if(HATCHING)
  list(APPEND FOLDERS hatching)
endif()
if(STABILIZED)
  list(APPEND FOLDERS stabilized)
endif()

set(LINK_LIBS AnalyzeProblem AnalyzeGeometric)
create_plato_analyze_library(AnalyzeElliptic "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "unittest"
              finite_deformation_mechanics/unittest) 
  if(HATCHING)
    list(APPEND FOLDERS hatching/unittest)
  endif()
  if(STABILIZED)
    list(APPEND FOLDERS stabilized/unittest)
  endif()

  set(LINK_LIBS AnalyzeTestUtilities AnalyzeGeometric) 
  create_plato_analyze_teuchos_unittester(AnalyzeElliptic "${FOLDERS}" "${LINK_LIBS}")
endif()
