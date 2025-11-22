include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)
include(${CMAKE_UTIL_DIR}/copy_test_data.cmake)

set(COPY_FILES  points.xyz)
copy_test_data("${COPY_FILES}")

set(FOLDERS ".")
add_subdirectory(./geometric)
if(ELLIPTIC)
  add_subdirectory(./elliptic)
endif()
if(PARABOLIC)
  add_subdirectory(./parabolic)
endif()
if(HELMHOLTZ)
  add_subdirectory(./helmholtz)
endif()
if(HYPERBOLIC)
  add_subdirectory(./hyperbolic)
endif()

set(LINK_LIBS AnalyzeUtilities 
              AnalyzeMesh 
              AnalyzeElement 
              AnalyzeSolver 
              AnalyzeDomain 
              AnalyzeMaterial 
              AnalyzeBoundaryConditions
              AnalyzeConstitutiveOperations 
              AnalyzeDifferentialOperations
              AnalyzeKinematicsOperations
              AnalyzeMappingOperations
              AnalyzeOptimizationOperations )
create_plato_analyze_header_library(AnalyzeProblem "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "unittest") 
  set(LINK_LIBS AnalyzeTestUtilities) 

  create_plato_analyze_teuchos_unittester(AnalyzeProblem "${FOLDERS}" "${LINK_LIBS}")
endif()
