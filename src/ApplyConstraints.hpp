#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Constrain all Dofs of given nodes
 * \param [in] aMatrix Matrix to be constrained
 * \param [in] aRhs    Vector to be constrained
 * \param [in] aNodes  List of node ids to be constrained
**********************************************************************************/
template<int NumDofPerNode>
void
applyBlockConstraints(
    Teuchos::RCP<Plato::CrsMatrixType>  aMatrix,
    Plato::ScalarVector                 aRhs,
    Plato::OrdinalVector                aNodes
)
/******************************************************************************/
{
    Plato::OrdinalType tNumNodes = aNodes.size();

    Plato::OrdinalVector tDofs("dof ids", tNumNodes * NumDofPerNode);

    Kokkos::parallel_for("dof ids", Kokkos::RangePolicy<>(0, tNumNodes), KOKKOS_LAMBDA(const Plato::OrdinalType & aNodeOrdinal)
    {
        for(Plato::OrdinalType tDof=0; tDof<NumDofPerNode; tDof++)
        {
            tDofs(aNodeOrdinal*NumDofPerNode + tDof) = aNodes(aNodeOrdinal)*NumDofPerNode+tDof;
        }
    });

    Plato::ScalarVector tVals("values", tDofs.extent(0));
    applyBlockConstraints<NumDofPerNode>(aMatrix, aRhs, tDofs, tVals, 1.0);
}
  
/******************************************************************************/
template<int NumDofPerNode>
void
applyBlockConstraints(
    Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
    Plato::ScalarVector                aRhs,
    Plato::OrdinalVector               aDirichletDofs,
    Plato::ScalarVector                aDirichletValues,
    Plato::Scalar                      aScale=1.0
  )
/******************************************************************************/
{
  
  /*
   Because both MAGMA and ViennaCL apparently assume symmetry even though it's technically
   not required for CG (and they do so in a way that breaks the solve badly), we do make the
   effort here to maintain symmetry while imposing BCs.
   */
    Plato::OrdinalType tNumBCs = aDirichletDofs.size();
    auto tRowMap        = aMatrix->rowMap();
    auto tColumnIndices = aMatrix->columnIndices();
    ScalarVector tMatrixEntries = aMatrix->entries();
    Kokkos::parallel_for("Dirichlet BC imposition - First loop", Kokkos::RangePolicy<>(0, tNumBCs), KOKKOS_LAMBDA(const Plato::OrdinalType & aBcOrdinal)
    {
        OrdinalType tRowDofOrdinal = aDirichletDofs[aBcOrdinal];
        Scalar tValue = aScale*aDirichletValues[aBcOrdinal];
        auto tRowNodeOrdinal = tRowDofOrdinal / NumDofPerNode;
        auto tLocalRowDofOrdinal  = tRowDofOrdinal % NumDofPerNode;
        OrdinalType tRowStart = tRowMap(tRowNodeOrdinal  );
        OrdinalType tRowEnd   = tRowMap(tRowNodeOrdinal+1);
        for (OrdinalType tColumnNodeOffset=tRowStart; tColumnNodeOffset<tRowEnd; tColumnNodeOffset++)
        {
            for (OrdinalType tLocalColumnDofOrdinal=0; tLocalColumnDofOrdinal<NumDofPerNode; tLocalColumnDofOrdinal++)
            {
                OrdinalType tColumnNodeOrdinal = tColumnIndices(tColumnNodeOffset);
                auto tEntryOrdinal = NumDofPerNode*NumDofPerNode*tColumnNodeOffset
                        + NumDofPerNode*tLocalRowDofOrdinal + tLocalColumnDofOrdinal;
                auto tColumnDofOrdinal = NumDofPerNode*tColumnNodeOrdinal+tLocalColumnDofOrdinal;
                if (tColumnDofOrdinal == tRowDofOrdinal) // diagonal
                {
                    tMatrixEntries(tEntryOrdinal) = 1.0;
                }
                else
                {
                    // correct the rhs to account for the fact that we'll be zeroing out (col,row) as well
                    // to maintain symmetry
                    Kokkos::atomic_add(&aRhs(tColumnDofOrdinal), -tMatrixEntries(tEntryOrdinal)*tValue);
                    tMatrixEntries(tEntryOrdinal) = 0.0;
                    OrdinalType tColRowStart = tRowMap(tColumnNodeOrdinal  );
                    OrdinalType tColRowEnd   = tRowMap(tColumnNodeOrdinal+1);
                    for (OrdinalType tColRowNodeOffset=tColRowStart; tColRowNodeOffset<tColRowEnd; tColRowNodeOffset++)
                    {
                        OrdinalType tColRowNodeOrdinal = tColumnIndices(tColRowNodeOffset);
                        auto tColRowEntryOrdinal = NumDofPerNode*NumDofPerNode*tColRowNodeOffset
                                +NumDofPerNode*tLocalColumnDofOrdinal+tLocalRowDofOrdinal;
                        auto tColRowDofOrdinal = NumDofPerNode*tColRowNodeOrdinal+tLocalRowDofOrdinal;
                        if (tColRowDofOrdinal == tRowDofOrdinal)
                        {
                            // this is the (col, row) entry -- clear it, too
                            tMatrixEntries(tColRowEntryOrdinal) = 0.0;
                        }
                    }
                }
            }
        }
    });
  
    Kokkos::parallel_for("Dirichlet BC imposition - Second loop", Kokkos::RangePolicy<int>(0,tNumBCs), KOKKOS_LAMBDA(int bcOrdinal){
        OrdinalType tDofOrdinal = aDirichletDofs[bcOrdinal];
        Scalar tValue = aScale*aDirichletValues[bcOrdinal];
        aRhs(tDofOrdinal) = tValue;
    });

}

/******************************************************************************/
template<int NumDofPerNode> void
applyConstraints(
    Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
    Plato::ScalarVector                aRhs,
    Plato::OrdinalVector               aNodes
)
/******************************************************************************/
{
    Plato::OrdinalType tNumNodes = aNodes.size();

    Plato::OrdinalVector tDofs("dof ids", tNumNodes * NumDofPerNode);

    Kokkos::parallel_for("dof ids", Kokkos::RangePolicy<>(0, tNumNodes), KOKKOS_LAMBDA(const Plato::OrdinalType & aNodeOrdinal)
    {
        for(Plato::OrdinalType tDof=0; tDof<NumDofPerNode; tDof++)
        {
            tDofs(aNodeOrdinal*NumDofPerNode + tDof) = aNodes(aNodeOrdinal)*NumDofPerNode+tDof;
        }
    });

    Plato::ScalarVector tVals("values", tDofs.extent(0));
    applyConstraints<NumDofPerNode>(aMatrix, aRhs, tDofs, tVals, 1.0);
}
  
/******************************************************************************/
template<int NumDofPerNode> void
applyConstraints(
    Teuchos::RCP<Plato::CrsMatrixType> matrix,
    Plato::ScalarVector                rhs,
    Plato::OrdinalVector               bcDofs,
    Plato::ScalarVector                bcValues,
    Plato::Scalar                      aScale=1.0
)
/******************************************************************************/
{
  
  /*
   Because both MAGMA and ViennaCL apparently assume symmetry even though it's technically
   not required for CG (and they do so in a way that breaks the solve badly), we do make the
   effort here to maintain symmetry while imposing BCs.
   */
  int numBCs = bcDofs.size();
  auto rowMap        = matrix->rowMap();
  auto columnIndices = matrix->columnIndices();
  ScalarVector matrixEntries = matrix->entries();
  Kokkos::parallel_for("BC imposition", Kokkos::RangePolicy<int>(0,numBCs), KOKKOS_LAMBDA(int bcOrdinal)
  {
    OrdinalType nodeNumber = bcDofs[bcOrdinal];
    Scalar value = aScale*bcValues[bcOrdinal];
    OrdinalType rowStart = rowMap(nodeNumber  );
    OrdinalType rowEnd   = rowMap(nodeNumber+1);
    for (OrdinalType entryOrdinal=rowStart; entryOrdinal<rowEnd; entryOrdinal++)
    {
      OrdinalType column = columnIndices(entryOrdinal);
      if (column == nodeNumber) // diagonal
      {
        matrixEntries(entryOrdinal) = 1.0;
      }
      else
      {
        // correct the rhs to account for the fact that we'll be zeroing out (col,row) as well
        // to maintain symmetry
        Kokkos::atomic_add(&rhs(column), -matrixEntries(entryOrdinal)*value);
        matrixEntries(entryOrdinal) = 0.0;
        OrdinalType colRowStart = rowMap(column  );
        OrdinalType colRowEnd   = rowMap(column+1);
        for (OrdinalType colRowEntryOrdinal=colRowStart; colRowEntryOrdinal<colRowEnd; colRowEntryOrdinal++)
        {
          OrdinalType colRowColumn = columnIndices(colRowEntryOrdinal);
          if (colRowColumn == nodeNumber)
          {
            // this is the (col, row) entry -- clear it, too
            matrixEntries(colRowEntryOrdinal) = 0.0;
          }
        }
      }
    }
  });
  
  Kokkos::parallel_for("BC imposition", Kokkos::RangePolicy<int>(0,numBCs), KOKKOS_LAMBDA(int bcOrdinal)
  {
    OrdinalType nodeNumber = bcDofs[bcOrdinal];
    Scalar value = aScale*bcValues[bcOrdinal];
    rhs(nodeNumber) = value;
  });
}

/******************************************************************************//**
 * \fn inline void apply_constraints
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Apply constraints to system of equations by modifying left and right hand sides.
 *
 * \param [in]     aBcDofs   degrees of freedom (dofs) associated with the boundary conditions
 * \param [in]     aBcValues scalar values forced at the dofs where the boundary conditions are applied
 * \param [in]     aScale    scalar multiplier
 * \param [in/out] aMatrix   left-hand-side matrix
 * \param [in/out] aRhs      right-hand-side vector
 *
 **********************************************************************************/
template<Plato::OrdinalType DofsPerNode>
inline void apply_constraints
(const Plato::OrdinalVector               & aBcDofs,
 const Plato::ScalarVector                & aBcValues,
 const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
       Plato::ScalarVector                & aRhs,
       Plato::Scalar                        aScale = 1.0)
{
    if(aMatrix->isBlockMatrix())
    {
        Plato::applyBlockConstraints<DofsPerNode>(aMatrix, aRhs, aBcDofs, aBcValues, aScale);
    }
    else
    {
        Plato::applyConstraints<DofsPerNode>(aMatrix, aRhs, aBcDofs, aBcValues, aScale);
    }
}
// function apply_constraints

/******************************************************************************//**
 * \fn inline void enforce_boundary_condition
 *
 * \brief Enforce boundary conditions.
 *
 * \param [in] aBcDofs    degrees of freedom associated with the boundary conditions
 * \param [in] aBcValues  values enforced in boundary degrees of freedom
 * \param [in/out] aState physical field
 *
 **********************************************************************************/
inline void
enforce_boundary_condition
(const Plato::OrdinalVector & aBcDofs,
 const Plato::ScalarVector  & aBcValues,
 const Plato::ScalarVector  & aState)
{
    auto tLength = aBcValues.size();
    Kokkos::parallel_for("enforce boundary condition", Kokkos::RangePolicy<>(0, tLength), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    {
        auto tDOF = aBcDofs(aOrdinal);
        aState(tDOF) = aBcValues(aOrdinal);
    });
}
// function enforce_boundary_condition

/******************************************************************************//**
 * \fn inline void set_dofs_values
 *
 * \brief Set values at degrees of freedom to input scalar (default scalar = 0.0).
 *
 * \param [in]     aBcDofs list of degrees of freedom (dofs)
 * \param [in]     aValue  scalar value (default = 0.0)
 * \param [in/out] aOutput output vector
 *
 **********************************************************************************/
inline void set_dofs_values
(const Plato::OrdinalVector & aBcDofs,
       Plato::ScalarVector  & aOutput,
       Plato::Scalar          aValue = 0.0)
{
    Kokkos::parallel_for("set values at bc dofs to zero", Kokkos::RangePolicy<>(0, aBcDofs.size()), KOKKOS_LAMBDA(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aBcDofs(aOrdinal)) = aValue;
    });
}
// function set_dofs_values

}
// namespace Plato
