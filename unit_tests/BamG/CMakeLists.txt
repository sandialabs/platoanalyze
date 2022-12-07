###############################################################################
# Sources:
###############################################################################
SET(BamG_SRCS BamG.cpp)

SET(BamG_HDRS BamG.hpp)

add_library(BamGlib ${BamG_SRCS} ${BamG_HDRS})

add_executable(BamG BamG_main.cpp)

set(BamG_LIBS Trilinos::all_selected_libs SEACASExodus::exodus)

target_link_libraries(BamGlib ${BamG_LIBS})
target_link_libraries(BamG BamGlib)

if( CMAKE_INSTALL_PREFIX )
  install( TARGETS BamG DESTINATION ${CMAKE_INSTALL_PREFIX}/bin )
endif()

###############################################################################
###############################################################################

add_subdirectory(./unittest)
