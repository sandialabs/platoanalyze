include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)

set(FOLDERS "."
            nonlinear_solvers)
set(LINK_LIBS AnalyzeCoreTypes 
              AnalyzeUtilities 
              AnalyzeLinearAlgebra 
              AnalyzeMesh 
              AnalyzeDomain 
              AnalyzeElement 
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

if(PLATOANALYZE_ENABLE_UMFPACK)
  message(STATUS "Adding define: -DPLATO_UMFPACK")
  add_definitions(-DPLATO_UMFPACK)

  find_library(UMFPACK NAMES umfpack PATHS ${UMFPACK_LIB_DIR})
  find_library(CHOLMOD NAMES cholmod PATHS ${UMFPACK_LIB_DIR})
  list(APPEND FOLDERS "umfpack")
  list(APPEND LINK_LIBS ${UMFPACK} ${CHOLMOD})
  list(APPEND EXTERNAL_INCLUDE_DIRECTORIES ${UMFPACK_INC_DIR})
endif()

if(PLATOANALYZE_ENABLE_TPETRA)
  list(APPEND FOLDERS "tpetra")
  list(APPEND LINK_LIBS TpetraCore::tpetra)
endif()

if(PLATOANALYZE_ENABLE_MESHMAP)
    list(APPEND FOLDERS "multipoint_constraint")
endif()

IF( AMGX_FOUND )
  list(APPEND FOLDERS "amgx")
  list(APPEND LINK_LIBS "${AMGX_LIBRARIES}")
ENDIF()

if (PLATOANALYZE_ENABLE_MESHMAP)
  list(APPEND LINK_LIBS ArborX::ArborX)
endif()

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
  if(PLATOANALYZE_ENABLE_UMFPACK)
    list(APPEND FOLDERS "umfpack/unittest")
  endif()
  set(LINK_LIBS AnalyzeTestUtilities) 
  create_plato_analyze_teuchos_unittester(AnalyzeSolver "${FOLDERS}" "${LINK_LIBS}")
  
endif()
