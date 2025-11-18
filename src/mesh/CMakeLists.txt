include(${CMAKE_UTIL_DIR}/create_plato_analyze_library.cmake)
include(${CMAKE_UTIL_DIR}/create_plato_analyze_unittester.cmake)
include(${CMAKE_UTIL_DIR}/copy_test_data.cmake)

set(COPY_FILES  hex-1-1-1.exo
                unit_cube_tet4.exo
                unit_cube_tet10.exo
                unit_cube_hex8.exo
                unit_cube_hex20.exo
                unit_cube_hex27.exo
                unit_cube_hex8_scalarField_gold.exo
                unit_cube_hex20_scalarField_gold.exo
                unit_cube_tet4_scalarField_gold.exo
                unit_cube_tet10_scalarField_gold.exo)
                
copy_test_data("${COPY_FILES}")

set(FOLDERS ".")
set(LINK_LIBS SEACASNemesis::nemesis 
              SEACASExodus::exodus 
              AnalyzeCoreTypes 
              AnalyzeParsing 
              AnalyzeLinearAlgebra 
              AnalyzeDomain 
              AnalyzeElement)

IF(OMEGA_H_PREFIX)
  list(APPEND LINK_LIBS Omega_h::omega_h)
  list(APPEND FOLDERS omega_h)
ENDIF()

if (PLATOANALYZE_ENABLE_MESHMAP)
  list(APPEND LINK_LIBS ArborX::ArborX)
endif()

create_plato_analyze_library(AnalyzeMesh "${FOLDERS}" "${LINK_LIBS}")

if( PLATOANALYZE_UNIT_TEST)
    set(LINK_LIBS AnalyzeTestUtilities)  
    set(FOLDERS unittest)

    create_plato_analyze_teuchos_unittester(AnalyzeMesh "${FOLDERS}" "${LINK_LIBS}")

endif()
