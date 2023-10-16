#include <PlatoTestHelpers.hpp>

#include <Teuchos_XMLParameterListHelpers.hpp>

#include <AnalyzeMacros.hpp>

namespace Plato {
namespace TestHelpers {

void
setControlWS
(std::vector<std::vector<Plato::Scalar>>& aValues,
 Plato::ScalarMultiVectorT<Plato::Scalar>& aControl)
{
    const auto tNumCells = aValues.size();
    assert(tNumCells <= aControl.extent(0));
    const auto tNumNodesPerCell = aControl.extent(1);
    assert(tNumNodesPerCell == aValues[0].size());

    std::vector<Plato::Scalar> tAllValues(tNumCells*tNumNodesPerCell);
    for(int iCell=0; iCell<tNumCells; iCell++)
        for(int iNode=0; iNode<tNumNodesPerCell; iNode++)
            tAllValues[iCell*tNumNodesPerCell+iNode] = aValues[iCell][iNode];

    auto tControlVals = Plato::TestHelpers::create_device_view(tAllValues);

    Kokkos::parallel_for("fill control", Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumCells), 
    KOKKOS_LAMBDA(Plato::OrdinalType iCell)
    {
        for(int iNode=0; iNode<tNumNodesPerCell; iNode++)
          aControl(iCell,iNode) = tControlVals(iCell*tNumNodesPerCell+iNode);
    });
}

auto get_box_mesh_with_spec(
    const std::string& aMeshType, 
    Plato::OrdinalType aMeshIntervals,
    const std::string& aFileName) -> std::tuple<Plato::Mesh, BamG::MeshSpec>{
  BamG::MeshSpec tSpec;
  tSpec.meshType = aMeshType;
  tSpec.fileName = aFileName;
  tSpec.numX = aMeshIntervals;
  tSpec.numY = aMeshIntervals;
  tSpec.numZ = aMeshIntervals;

  BamG::generate(tSpec);

  return {Plato::MeshFactory::create(tSpec.fileName), tSpec};
}

Plato::Mesh get_box_mesh(std::string aMeshType, Plato::OrdinalType aMeshIntervals,
                         std::string aFileName) {
    return std::get<0>(get_box_mesh_with_spec(aMeshType, aMeshIntervals, aFileName));
}

Plato::Mesh get_box_mesh(std::string aMeshType, Plato::Scalar aMeshWidthX,
                         Plato::OrdinalType aMeshIntervalsX,
                         Plato::Scalar aMeshWidthY,
                         Plato::OrdinalType aMeshIntervalsY,
                         Plato::Scalar aMeshWidthZ,
                         Plato::OrdinalType aMeshIntervalsZ) {
  BamG::MeshSpec tSpec;
  tSpec.meshType = aMeshType;
  tSpec.fileName = "BamG_unit_test_mesh.exo";
  tSpec.numX = aMeshIntervalsX;
  tSpec.numY = aMeshIntervalsY;
  tSpec.numZ = aMeshIntervalsZ;
  tSpec.dimX = aMeshWidthX;
  tSpec.dimY = aMeshWidthY;
  tSpec.dimZ = aMeshWidthZ;

  BamG::generate(tSpec);

  return Plato::MeshFactory::create(tSpec.fileName);
}

void set_dof_value_in_vector_on_boundary_2D(
    Plato::Mesh aMesh, const std::string &aBoundaryID,
    const Plato::ScalarVector &aDofValues, const Plato::OrdinalType &aDofStride,
    const Plato::OrdinalType &aDofToSet, const Plato::Scalar &aSetValue) {
  auto tBoundaryNodes = aMesh->GetNodeSetNodes(aBoundaryID);

  auto tNumBoundaryNodes = tBoundaryNodes.size();

  Kokkos::parallel_for("fill vector boundary dofs",
      Kokkos::RangePolicy<>(0, tNumBoundaryNodes),
      KOKKOS_LAMBDA(const Plato::OrdinalType &aIndex) {
        Plato::OrdinalType tIndex =
            aDofStride * tBoundaryNodes(aIndex) + aDofToSet;
        aDofValues(tIndex) += aSetValue;
      });
}

void set_dof_value_in_vector_on_boundary_3D(
    Plato::Mesh aMesh, const std::string &aBoundaryID,
    const Plato::ScalarVector &aDofValues, const Plato::OrdinalType &aDofStride,
    const Plato::OrdinalType &aDofToSet, const Plato::Scalar &aSetValue) {
  auto tLocalOrdinals = aMesh->GetNodeSetNodes(aBoundaryID);
  auto tNumBoundaryNodes = tLocalOrdinals.size();

  Kokkos::parallel_for("fill vector boundary dofs",
      Kokkos::RangePolicy<>(0, tNumBoundaryNodes),
      KOKKOS_LAMBDA(const Plato::OrdinalType &aIndex) {
        Plato::OrdinalType tIndex =
            aDofStride * tLocalOrdinals(aIndex) + aDofToSet;
        aDofValues(tIndex) += aSetValue;
      });
}

Plato::OrdinalVector get_dirichlet_indices_on_boundary_2D(
    Plato::Mesh aMesh, const std::string &aBoundaryID,
    const Plato::OrdinalType &aDofStride, const Plato::OrdinalType &aDofToSet) {
  auto tBoundaryNodes = aMesh->GetNodeSetNodes(aBoundaryID);
  Plato::OrdinalVector tDofIndices;
  auto tNumBoundaryNodes = tBoundaryNodes.size();
  Kokkos::resize(tDofIndices, tNumBoundaryNodes);

  Kokkos::parallel_for("fill dirichlet dof indices on boundary 2-D",
      Kokkos::RangePolicy<>(0, tNumBoundaryNodes),
      KOKKOS_LAMBDA(const Plato::OrdinalType &aIndex) {
        Plato::OrdinalType tIndex =
            aDofStride * tBoundaryNodes[aIndex] + aDofToSet;
        tDofIndices(aIndex) = tIndex;
      });

  return (tDofIndices);
}

Plato::OrdinalVector get_dirichlet_indices_on_boundary_3D(
    Plato::Mesh aMesh, const std::string &aBoundaryID,
    const Plato::OrdinalType &aDofStride, const Plato::OrdinalType &aDofToSet) {
  auto tBoundaryNodes = aMesh->GetNodeSetNodes(aBoundaryID);
  Plato::OrdinalVector tDofIndices;
  auto tNumBoundaryNodes = tBoundaryNodes.size();
  Kokkos::resize(tDofIndices, tNumBoundaryNodes);

  Kokkos::parallel_for("fill dirichlet dof indices on boundary 3-D",
      Kokkos::RangePolicy<>(0, tNumBoundaryNodes),
      KOKKOS_LAMBDA(const Plato::OrdinalType &aIndex) {
        Plato::OrdinalType tIndex =
            aDofStride * tBoundaryNodes[aIndex] + aDofToSet;
        tDofIndices(aIndex) = tIndex;
      });

  return (tDofIndices);
}

void set_dof_value_in_vector(const Plato::ScalarVector &aDofValues,
                             const Plato::OrdinalType &aDofStride,
                             const Plato::OrdinalType &aDofToSet,
                             const Plato::Scalar &aSetValue) {
  auto tVectorSize = aDofValues.extent(0);
  auto tRange = tVectorSize / aDofStride;

  Kokkos::parallel_for("fill specific vector entry globally",
      Kokkos::RangePolicy<>(0, tRange),
      KOKKOS_LAMBDA(const Plato::OrdinalType &aNodeIndex) {
        Plato::OrdinalType tIndex = aDofStride * aNodeIndex + aDofToSet;
        aDofValues(tIndex) += aSetValue;
      });
}

std::vector<std::vector<Plato::Scalar>>
to_full(Teuchos::RCP<Plato::CrsMatrixType> aInMatrix) {
  using Plato::OrdinalType;
  using Plato::Scalar;

  std::vector<std::vector<Scalar>> retMatrix(
      aInMatrix->numRows(), std::vector<Scalar>(aInMatrix->numCols(), 0.0));

  const auto tNumRowsPerBlock = aInMatrix->numRowsPerBlock();
  const auto tNumColsPerBlock = aInMatrix->numColsPerBlock();
  const auto tBlockSize = tNumRowsPerBlock * tNumColsPerBlock;

  const auto tRowMap = get(aInMatrix->rowMap());
  const auto tColMap = get(aInMatrix->columnIndices());
  const auto tValues = get(aInMatrix->entries());

  const auto tNumRows = tRowMap.extent(0) - 1;
  for (OrdinalType iRowIndex = 0; iRowIndex < tNumRows; iRowIndex++) {
    const auto tFrom = tRowMap(iRowIndex);
    const auto tTo = tRowMap(iRowIndex + 1);
    for (auto iColMapEntryIndex = tFrom; iColMapEntryIndex < tTo;
         iColMapEntryIndex++) {
      const auto tBlockColIndex = tColMap(iColMapEntryIndex);
      for (OrdinalType iLocalRowIndex = 0; iLocalRowIndex < tNumRowsPerBlock;
           iLocalRowIndex++) {
        const auto tRowIndex = iRowIndex * tNumRowsPerBlock + iLocalRowIndex;
        for (OrdinalType iLocalColIndex = 0; iLocalColIndex < tNumColsPerBlock;
             iLocalColIndex++) {
          const auto tColIndex =
              tBlockColIndex * tNumColsPerBlock + iLocalColIndex;
          const auto tSparseIndex = iColMapEntryIndex * tBlockSize +
                                    iLocalRowIndex * tNumColsPerBlock +
                                    iLocalColIndex;
          retMatrix[tRowIndex][tColIndex] = tValues[tSparseIndex];
        }
      }
    }
  }
  return retMatrix;
}

const Teuchos::RCP<Teuchos::ParameterList> getParameterListForHelmholtzTest() {
  return
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>"
    "  <ParameterList name='Spatial Model'>"
    "    <ParameterList name='Domains'>"
    "      <ParameterList name='Design Volume'>"
    "        <Parameter name='Element Block' type='string' value='body'/>"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>"
    "      </ParameterList>"
    "    </ParameterList>"
    "  </ParameterList>"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/>"
    "  <Parameter name='Physics' type='string' value='Helmholtz Filter'/>"
    "  <ParameterList name='Parameters'>"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>"
    "  </ParameterList>"
    "</ParameterList>"
  );
}

const Teuchos::RCP<Teuchos::ParameterList> getSolverParametersForHelmholtzTest() {
  return 
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>"
    "</ParameterList>"
  );
}
} // namespace TestHelpers
} // namespace Plato
