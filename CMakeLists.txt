cmake_minimum_required(VERSION 3.21)
CMAKE_POLICY(SET CMP0057 NEW)

project(PLATOANALYZE VERSION 0.0.0 LANGUAGES CXX)

# hdf5 dependency (introduced by exodus) needs C
enable_language(C)

set (CMAKE_CXX_STANDARD 20)

option( WARNINGS_AS_ERRORS "Builds platoanalyze with warnings as errors" OFF )
option( BUILD_WITH_SANITIZER_FLAGS "Builds platoanalyze with sanitizers" OFF )
option(PLATOANALYZE_ENABLE_CUDA "Build for Nvidia GPUs using CUDA" OFF)
option(PLATOANALYZE_ENABLE_MESHMAP  "Enable Plato MeshMap"  ON )
option(PLATOANALYZE_ENABLE_ENGINEMESH "Enable EngineMesh"           ON )
option(ELLIPTIC        "Compile with Elliptic physics"       ON )
option(PARABOLIC       "Compile with Parabolic physics"      ON )
option(HYPERBOLIC      "Compile with Hyperbolic physics"     OFF )
option(CUSTOMMATERIALS "Compile with custom materials"       OFF)
option(PLATOANALYZE_UNIT_TEST   "Flag to enable unit tests"  ON)
option(PLATOANALYZE_SMOKE_TESTS "Flag to enable smoke tests" ON)
option(PLATOANALYZE_LONG_LONG_ORDINALTYPE "Flag to change Plato::OrdinalType to 'long long'" OFF)
option( PLATOANALYZE_CHECK_FPE "Flag to enable Plato Analyze checks for floating point exceptions" OFF)

MESSAGE("\n\nConfiguring Plato Analyze Version ${PLATOANALYZE_VERSION_MAJOR}.${PLATOANALYZE_VERSION_MINOR}.${PLATOANALYZE_VERSION_PATCH}")

find_program(CCACHE_FOUND ccache)
if( CCACHE_FOUND )
  # Spack is able to use ccache, however, using ccache with nvcc_wrapper does not seem to work 
  # properly through spack.
  message( STATUS "Found ccache" )
  set( CMAKE_CXX_COMPILER_LAUNCHER ccache )
endif()

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  file(CREATE_LINK "${CMAKE_BINARY_DIR}/compile_commands.json" "${CMAKE_SOURCE_DIR}/compile_commands.json" SYMBOLIC )
endif()

IF (DEFINED Trilinos_PREFIX)
  SET(CMAKE_PREFIX_PATH ${Trilinos_PREFIX} ${CMAKE_PREFIX_PATH})
ENDIF()

#silences warnings about the system not supporting shared libraries
#http://stackoverflow.com/questions/12264299/cmake-on-linux-target-platform-does-not-support-dynamic-linking
SET_PROPERTY(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS TRUE)

FIND_PACKAGE(MPI REQUIRED)

# Get Trilinos as one entity
FIND_PACKAGE(Trilinos REQUIRED)

IF(NOT Trilinos_FOUND)
  MESSAGE(FATAL_ERROR "Cannot find Trilinos - build exiting!!!")
ENDIF()

# Echo trilinos build info just for fun
MESSAGE("\nFound Trilinos!  Here are the details: ")
MESSAGE("   Trilinos_PREFIX = ${Trilinos_PREFIX}")
MESSAGE("   Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("   Trilinos_VERSION = ${Trilinos_VERSION}")
MESSAGE("   Trilinos_CXX_COMPILER = ${Trilinos_CXX_COMPILER}")
MESSAGE("   Trilinos_CXX_COMPILER_FLAGS = ${Trilinos_CXX_COMPILER_FLAGS}")
MESSAGE("End of Trilinos details\n")

find_package(Boost REQUIRED)

message(STATUS "Additional gcc warnings: ${GCC_ADDITIONAL_WARNINGS}")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:${GCC_ADDITIONAL_WARNINGS}>")

message(STATUS "Additional clang warnings: ${CLANG_ADDITIONAL_WARNINGS}")
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:${CLANG_ADDITIONAL_WARNINGS}>")

if(WARNINGS_AS_ERRORS)
  message(STATUS "Using -Werror")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Werror>")
endif()

if(BUILD_WITH_SANITIZER_FLAGS)
  message(STATUS "Compiling with sanitizer flags turned on")
  message(STATUS "Sanitizer flags: ${SANITIZER_FLAGS}")
  add_compile_options(${SANITIZER_FLAGS})
  add_link_options(${SANITIZER_FLAGS})
endif()

## this has to be conditional, or building without cuda fails
if (PLATOANALYZE_ENABLE_CUDA)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
ENDIF ()

MESSAGE("-- CXX Flags being used in this build are : ${CMAKE_CXX_FLAGS}")
if( ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo" AND PLATOANALYZE_ENABLE_CUDA )
  # PA has several tests that are seg-faulting with O2, switch to O1
  STRING(REPLACE "O2" "O1" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
  MESSAGE(STATUS "Modifying CXX build specific flags : ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release" AND PLATOANALYZE_ENABLE_CUDA )
  # PA has several tests that are seg-faulting with O2, switch to O1
  STRING(REPLACE "O3" "O1" CMAKE_CXX_FLAGS_RELEASE ${CMAKE_CXX_FLAGS_RELEASE})
  MESSAGE(STATUS "Modifying CXX build specific flags : ${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# disallow in-source builds

get_filename_component(srcdir "${CMAKE_SOURCE_DIR}" REALPATH)

get_filename_component(bindir "${CMAKE_BINARY_DIR}" REALPATH)

if(${srcdir} STREQUAL ${bindir})

    message("######################################################")

    message("You are attempting to build in your Source Directory.")

    message("You must run cmake from a build directory.")

    message("######################################################")

    message(FATAL_ERROR "In-source builds are forbidden!")

endif()

SET(OMEGA_H_PREFIX "" CACHE PATH "Omega_h install location")
IF(OMEGA_H_PREFIX)
  SET(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${OMEGA_H_PREFIX})
  FIND_PACKAGE(Omega_h 9.17.0 REQUIRED)
ENDIF()

# DAI: omega_h2 has a vtkdiff which by default compares all timesteps.
# old omega_h's vtkdiff only compared one timestep.
# at the time of transition, we will keep comparing just one time step
# this flag is needed to do that.

# We need to do this better, but for now lets see how well it works to find the right mpirun
# This needs to be the one on the front end for Cuda and MIC machines - it runs the utilities.

IF(PLATOANALYZE_MPIEXEC)
  MESSAGE("-- User defined PLATOANALYZE_MPIEXEC=${PLATOANALYZE_MPIEXEC}")
  SEPARATE_ARGUMENTS(PLATOANALYZE_MPIEXEC)
ELSE()
  SET(PLATOANALYZE_MPIEXEC)

  #mpirun is frequently found alongside mpicxx, try looking for it there.
  get_filename_component(COMPILER_DIR ${CMAKE_CXX_COMPILER} PATH)

  FIND_FILE(
    PLATOANALYZE_MPIEXEC
    NAMES mpirun mpiexec
    PATHS "${COMPILER_DIR}" ENV PATH
    )

  IF(PLATOANALYZE_MPIEXEC)
    MESSAGE("-- Found PLATOANALYZE_MPIEXEC=${PLATOANALYZE_MPIEXEC}")
  ELSE()
    MESSAGE(FATAL_ERROR "Fatal Error: Cannot find mpirun or mpiexec for PLATOANALYZE_MPIEXEC")
  ENDIF()
ENDIF()

add_definitions(-DPLATOANALYZE_USE_EXPLICIT_INSTANTIATION)

if ( PLATOANALYZE_CHECK_FPE)
  add_definitions(-DPLATOANALYZE_CHECK_FPE)
  if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    add_definitions(-DPLATOANALYZE_USE_GNU_FPE)
    message("-- Building with GNU FPE checking enabled.")
  else()
    message("-- Building with XMM FPE checking enabled.")
  endif()
else()
  message("-- Building without FPE checking.")
endif()

if(PLATOANALYZE_LONG_LONG_ORDINALTYPE)
  add_definitions(-DPLATOANALYZE_LONG_LONG_ORDINALTYPE)
  message("-- Building with 'long long' ordinal type.")
endif()

add_executable(
    analyze
    src/main/Run.cpp
    src/main/Analyze.cpp
  )

if( CMAKE_INSTALL_PREFIX )
  install( TARGETS analyze DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )
endif()
target_include_directories(analyze PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries( analyze AnalyzeLib)

IF (DEFINED AMGX_PREFIX)
  ADD_DEFINITIONS(-DHAVE_AMGX)
  FIND_PATH(AMGX_INCLUDE_DIR NAMES amgx_c.h PATHS ${AMGX_PREFIX}/include)
  IF (NOT AMGX_INCLUDE_DIR)
    MESSAGE(FATAL_ERROR "Couldn't find AMGX header!")
  ENDIF()
  FIND_LIBRARY(AMGX_LIB NAMES amgxsh amgx PATHS ${AMGX_PREFIX}/lib)
  IF (NOT AMGX_LIB)
    MESSAGE(FATAL_ERROR "Couldn't find AMGX library!")
  ENDIF()
  IF(PLATOANALYZE_LONG_LONG_ORDINALTYPE)
    message(FATAL_ERROR "Can't build with AMGX when building with 'long long' ordinal type. \n
    Please set -DPLATOANALYZE_LONG_LONG_ORDINALTYPE=OFF or unset AMGX_PREFIX")
  ENDIF()
  SET(AMGX_INCLUDE_DIRS "${AMGX_INCLUDE_DIR}")
  SET(AMGX_LIBRARIES "${AMGX_LIB}")
  SET(AMGX_FOUND TRUE) # a bit of a hack; would be better to check that it actually exists!
  MESSAGE("-- Found AmgX!")
  MESSAGE("-- AMGX_INCLUDE_DIRS = ${AMGX_INCLUDE_DIRS}")
  MESSAGE("-- AMGX_LIBRARIES = ${AMGX_LIBRARIES}")
ENDIF()

if(ELLIPTIC)
  message("-- Plato: Elliptic physics added")
  ADD_DEFINITIONS(-DPLATO_ELLIPTIC)
endif()
if(PARABOLIC)
  message("-- Plato: Parabolic physics added")
  ADD_DEFINITIONS(-DPLATO_PARABOLIC)
endif()
if(HYPERBOLIC)
  message("-- Plato: Hyperbolic physics added")
  ADD_DEFINITIONS(-DPLATO_HYPERBOLIC)
endif()

if(CUSTOMMATERIALS)
  message("-- Plato: Custom materials added")
  ADD_DEFINITIONS(-DPLATO_CUSTOM_MATERIALS)
endif()

if (PLATOANALYZE_ENABLE_MESHMAP)
  add_definitions(-DPLATO_MESHMAP)
  find_package(ArborX REQUIRED)
endif()

IF(PLATOANALYZE_ENABLE_ENGINEMESH)
  FIND_PROGRAM( EXODIFF NAMES exodiff PATHS ${Trilinos_PREFIX}/bin NO_DEFAULT_PATH)
  IF(EXODIFF)
    MESSAGE("-- EXODIFF=${EXODIFF}")
  ELSE()
    MESSAGE(FATAL_ERROR "Cannot locate the executable for exodiff, exiting")
  ENDIF()
ELSE()
  FIND_PROGRAM( VTKDIFF NAMES vtkdiff PATHS ${OMEGA_H_PREFIX}/bin NO_DEFAULT_PATH)
  IF(VTKDIFF)
    MESSAGE("-- VTKDIFF=${VTKDIFF}")
  ELSE()
    MESSAGE(FATAL_ERROR "Cannot locate the executable for vtkdiff, exiting")
  ENDIF()
ENDIF()


find_package(Tpetra REQUIRED)
find_package(Belos REQUIRED)
find_package(Ifpack2 REQUIRED)


set(PLATOENGINE_PREFIX "" CACHE STRING "Path to Plato install")
find_package(PlatoEngine REQUIRED COMPONENTS PlatoEngine::PlatoCommonInterface PlatoEngine::PlatoUtilitiesInterface PlatoEngine::PlatoTestUtilitiesInterface PlatoEngine::PlatoCommonTestUtilitiesInterface HINTS ${PLATOENGINE_PREFIX})
find_package(SEACASExodus REQUIRED)
find_package(SEACASNemesis REQUIRED)
find_package(Shards REQUIRED)
find_package(TeuchosCore REQUIRED)

if (PLATOANALYZE_ENABLE_ENGINEMESH)
  message("Default mesh is EngineMesh")
else()
  add_definitions(-DUSE_OMEGAH_MESH)
  message("Default mesh is Omega-h mesh")
endif()

option(PLATOANALYZE_ENABLE_TACHO "Enable Tacho" OFF)
option(PLATOANALYZE_ENABLE_UMFPACK "Enable UMFPACK" OFF)

IF(OMEGA_H_PREFIX)
 # omega-h adds -Werror which currently breaks build
 set_target_properties(Omega_h::omega_h PROPERTIES INTERFACE_COMPILE_OPTIONS "")
ENDIF()


SET(PLATOANALYZE_RANKS_PER_NODE_FLAG "-npernode" CACHE STRING "Processes per node flag for PLATOANALYZE_MPIEXEC")
SET(PLATOANALYZE_RANKS_FLAG "-n" CACHE STRING "Number of processes flag for PLATOANALYZE_MPIEXEC")
SET(PLATOANALYZE_RANKS_PER_NODE "1" CACHE STRING "Maximum number of processes per node")
SET(PLATOANALYZE_VALGRIND "" CACHE STRING "Optional Valgrind call and arguments")

function(build_mpi_test_string RESULT_NAME num_mpi_proc)
  string(REPLACE " " ";" VALGRIND "${PLATOANALYZE_VALGRIND}")
  set(${RESULT_NAME} ${PLATOANALYZE_MPIEXEC} ${PLATOANALYZE_RANKS_PER_NODE_FLAG} ${PLATOANALYZE_RANKS_PER_NODE} ${PLATOANALYZE_RANKS_FLAG} ${num_mpi_proc} ${VALGRIND} ${ARGN} PARENT_SCOPE)
endfunction(build_mpi_test_string)

ENABLE_TESTING()
INCLUDE(CTest)

add_subdirectory(src)

if(PLATOANALYZE_SMOKE_TESTS)
  add_subdirectory(tests)
  add_subdirectory(python)
endif()

set(INTEGRATION_TESTS_DIR ${PROJECT_SOURCE_DIR}/../plato-integration-tests ) # Assumes we cloned from the plato super-project
if(PLATOANALYZE_INTEGRATION_TESTS)
  add_subdirectory(${INTEGRATION_TESTS_DIR}/analyze ${PROJECT_BINARY_DIR}/integration_tests/analyze)
endif()

set(PLATOANALYZE_KEY_INTS
    PLATOANALYZE_VERSION_MAJOR
    PLATOANALYZE_VERSION_MINOR
    PLATOANALYZE_VERSION_PATCH
   )
