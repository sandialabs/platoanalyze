include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS "."
            nonlinear_solvers
            multipoint_constraint)
set(LINK_LIBS AnalyzeCoreTypes 
              AnalyzeUtilities 
              AnalyzeLinearAlgebra 
              AnalyzeMesh 
              AnalyzeDomain 
              AnalyzeElement 
              ArborX::ArborX
              Belos::belos 
              MueLu::muelu)

set(EXTERNAL_INCLUDE_DIRECTORIES "")

if(PLATOANALYZE_ENABLE_TACHO)
  message(STATUS "Adding define: -DPLATO_TACHO")
  add_definitions(-DPLATO_TACHO)

  find_package(ShyLU_NodeTacho REQUIRED)
  list(APPEND FOLDERS "tacho")
  list(APPEND LINK_LIBS Shards::shards  $<$<BOOL:${PLATOANALYZE_ENABLE_CUDA}>:CUDA::cusparse> # This is a work-around for tacho not linking cuda properly.
                  ${ShyLU_NodeTacho_LIBRARIES})
endif()

if(PLATOANALYZE_ENABLE_SUITESPARSE)
  message(STATUS "Adding define: -DPLATO_SUITESPARSE")
  add_definitions(-DPLATO_SUITESPARSE)

  find_library(UMFPACK NAMES umfpack PATHS ${SUITESPARSE_LIB_DIR})
  find_library(CHOLMOD NAMES cholmod PATHS ${SUITESPARSE_LIB_DIR})
  list(APPEND FOLDERS "suitesparse")
  list(APPEND LINK_LIBS ${UMFPACK} ${CHOLMOD})
  list(APPEND EXTERNAL_INCLUDE_DIRECTORIES ${SUITESPARSE_INC_DIR})
endif()


list(APPEND FOLDERS "tpetra")
list(APPEND LINK_LIBS TpetraCore::tpetra)

IF( AMGX_FOUND )
  list(APPEND FOLDERS "amgx")
  list(APPEND LINK_LIBS "${AMGX_LIBRARIES}")
ENDIF()

IF( AMGX_FOUND )
  list(APPEND FOLDERS ${AMGX_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/src/solverInterfaces/AmgX/)
ENDIF()

create_plato_analyze_library(AnalyzeSolver "${FOLDERS}" "${LINK_LIBS}")
target_include_directories(AnalyzeSolver PUBLIC "${EXTERNAL_INCLUDE_DIRECTORIES}")

if( PLATOANALYZE_UNIT_TEST )
  set(FOLDERS "")
  if(PLATOANALYZE_ENABLE_TACHO)
    list(APPEND FOLDERS "tacho/unittest")
  endif()
  if(PLATOANALYZE_ENABLE_SUITESPARSE)
    list(APPEND FOLDERS "suitesparse/unittest")
  endif()
  set(LINK_LIBS AnalyzeTestUtilities) 
  create_plato_analyze_teuchos_unittester(AnalyzeSolver "${FOLDERS}" "${LINK_LIBS}")
  
endif()
