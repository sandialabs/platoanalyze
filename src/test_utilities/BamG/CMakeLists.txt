include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)


set(LINK_LIBS SEACASExodus::exodus 
              TeuchosCore::teuchoscore)
create_plato_analyze_library(AnalyzeBamG "." "${LINK_LIBS}")

add_executable(BamG main/BamG_main.cpp)

target_link_libraries(BamG AnalyzeBamG)

if( CMAKE_INSTALL_PREFIX )
  install( TARGETS BamG DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )
endif()

###############################################################################
###############################################################################

add_subdirectory(./unittest)
