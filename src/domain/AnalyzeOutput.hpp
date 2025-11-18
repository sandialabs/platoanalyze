#ifndef PLATO_OUTPUT_HPP
#define PLATO_OUTPUT_HPP

#include <Teuchos_ParameterList.hpp>
#include <string>

#include "domain/Solutions.hpp"
#include "mesh/PlatoMesh.hpp"
#include "utilities/PlatoUtilities.hpp"
namespace Plato
{
void AddStateData(Plato::MeshIO aWriter, Plato::DataMap aDataMap, Plato::OrdinalType aNumDims);

/******************************************************************************/ /**
                                                                                  * \brief Output data for all your
                                                                                  *output needs \param [in]
                                                                                  *aOutputFilePath  output viz file path
                                                                                  * \param [in] aSolutionsOutput global
                                                                                  *solution data for output \param [in]
                                                                                  *aStateDataMap    Plato Analyze data
                                                                                  *map \param [in] aMesh            mesh
                                                                                  *database
                                                                                  **********************************************************************************/
void universal_solution_output(const std::string& aOutputFilePath,
                               const Plato::Solutions& aSolutionsOutput,
                               const Plato::DataMap& aStateDataMap,
                               Plato::Mesh aMesh);
}  // namespace Plato
// namespace Plato

#endif
