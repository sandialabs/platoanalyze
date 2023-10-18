#include "util/PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_ParameterList.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "helmholtz/Helmholtz.hpp"
#include "helmholtz/VectorFunction.hpp"
#include "helmholtz/HelmholtzElement.hpp"
#include "helmholtz/Problem.hpp"

#include "Tri3.hpp"
#include "Tet4.hpp"
#include "PlatoMathHelpers.hpp"
#include "alg/PlatoSolverFactory.hpp"

TEUCHOS_UNIT_TEST( HelmholtzFilterTests, Tri3MatrixEntries )
{
  // create test mesh
  //
  constexpr int meshWidth=1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("TRI3", meshWidth);

  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Tri3>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  Plato::DataMap tDataMap;
  const auto tParamList = Plato::TestHelpers::getParameterListForHelmholtzTest();
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);
  Plato::Helmholtz::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

/*

  Matlab code for generating the gold data

points = [0 0
          0 1
          1 0
          1 1];

triangles = [1 3 4;
             1 4 2];

locM = [2 1 1; 1 2 1; 1 1 2]/24;

nnodes = size(points,1);
nelems = size(triangles,1);

M = zeros(nnodes);
K = zeros(nnodes);

for e=1:nelems
    ind = triangles(e,:);
    X = points(ind,:)';
    J = [X(:,2) - X(:,1), X(:,3) - X(:,1)];
    dJ = det(J);
    
    M(ind,ind) = M(ind,ind) + dJ*locM;
    
    D = [-1 -1; 1 0; 0 1]*inv(J);
    
    locK = D*D';
    
    K(ind,ind) = K(ind,ind) + dJ/2*locK;
end

r = 0.1;
ans = r^2*K+M;

*/

  namespace pth = Plato::TestHelpers;

  const std::vector<Plato::OrdinalType> tRowMapA = {0, 4, 7, 10, 14};
  const std::vector<Plato::OrdinalType> tColIndices = {0, 1, 2, 3,
                                                       0, 1, 3,
                                                       0, 2, 3,
                                                       0, 1, 2, 3};
  const std::vector<Plato::Scalar>      tValuesA = {0.1766666666666667, 0.0366666666666667, 0.0366666666666667, 0.0833333333333333,
                                                    0.0366666666666667, 0.0933333333333333, /* 0.0000000000, */ 0.0366666666666667,
                                                    0.0366666666666667, /* 0.0000000000, */ 0.0933333333333333, 0.0366666666666667,
                                                    0.0833333333333333, 0.0366666666666667, 0.0366666666666667, 0.1766666666666667};

  auto tHostRowMap = Kokkos::create_mirror_view(jacobian->rowMap());
  auto tHostColumnIndices = Kokkos::create_mirror_view(jacobian->columnIndices());
  auto tHostValues = Kokkos::create_mirror_view(jacobian->entries());
  Kokkos::deep_copy(tHostRowMap, jacobian->rowMap());
  Kokkos::deep_copy(tHostColumnIndices, jacobian->columnIndices());
  Kokkos::deep_copy(tHostValues, jacobian->entries());

  TEST_COMPARE_ARRAYS(tHostRowMap, tRowMapA);
  TEST_COMPARE_ARRAYS(tHostColumnIndices, tColIndices);
  TEST_COMPARE_FLOATING_ARRAYS(tHostValues, tValuesA, 1e-13);
}

TEUCHOS_UNIT_TEST( HelmholtzFilterTests, Tet4MatrixEntries )
{
  // create test mesh
  //
  constexpr int meshWidth=1;
  auto tMesh = Plato::TestHelpers::get_box_mesh("Tet4", meshWidth);

  using PhysicsType = ::Plato::HelmholtzFilter<Plato::Tet4>;
  using ElementType = typename PhysicsType::ElementType;

  int tNumDofsPerNode = ElementType::mNumDofsPerNode;
  int tNumNodes = tMesh->NumNodes();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  Plato::DataMap tDataMap;
  const auto tParamList = Plato::TestHelpers::getParameterListForHelmholtzTest();
  Plato::SpatialModel tSpatialModel(tMesh, *tParamList, tDataMap);
  Plato::Helmholtz::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

/*

  Matlab code for generating the gold data

locM = [2 1 1 1; 1 2 1 1; 1 1 2 1; 1 1 1 2]/120;

M = zeros(nnodes);
K = zeros(nnodes);

for e=1:nelems
    ind = blk01(:,e);
    X = [x0(ind) y0(ind) z0(ind)]';
    J = [X(:,2) - X(:,1), X(:,3) - X(:,1), X(:,4) - X(:,1)];
    dJ = det(J);
    
    M(ind,ind) = M(ind,ind) + dJ*locM;
    
    D = [-1 -1 -1; 1 0 0; 0 1 0; 0 0 1]*inv(J);
    
    locK = D*D';
    
    K(ind,ind) = K(ind,ind) + dJ/6*locK;
end

r = 0.1;
ans = r^2*K+M;

*/

  namespace pth = Plato::TestHelpers;

  const std::vector<Plato::OrdinalType> tRowMapA = {0, 8, 13, 18, 23, 28, 33, 38, 46};
  const std::vector<Plato::OrdinalType> tColIndices = {0, 1, 2, 3, 4, 5, 6, 7,
                                                       0, 1, 3, 5, 7,
                                                       0, 2, 3, 6, 7,
                                                       0, 1, 2, 3, 7,
                                                       0, 4, 5, 6, 7,
                                                       0, 1, 4, 5, 7,
                                                       0, 2, 4, 6, 7,
                                                       0, 1, 2, 3, 4, 5, 6, 7};
  const std::vector<Plato::Scalar>      tValuesA = {
    0.110000000000000, 0.013333333333333, 0.013333333333333, 0.016666666666667, 0.013333333333333, 0.016666666666667, 0.016666666666667, 0.050000000000000,
    0.013333333333333, 0.040000000000000, /* 0.000000000 */  0.006666666666667, /* 0.000000000 */  0.006666666666667, /* 0.000000000 */  0.016666666666667,
    0.013333333333333, /* 0.000000000 */  0.040000000000000, 0.006666666666667, /* 0.000000000 */  /* 0.000000000 */  0.006666666666667, 0.016666666666667,
    0.016666666666667, 0.006666666666667, 0.006666666666667, 0.040000000000000, /* 0.000000000 */  /* 0.000000000 */  /* 0.000000000 */  0.013333333333333,
    0.013333333333333, /* 0.000000000 */  /* 0.000000000 */  /* 0.000000000 */  0.040000000000000, 0.006666666666667, 0.006666666666667, 0.016666666666667,
    0.016666666666667, 0.006666666666667, /* 0.000000000 */  /* 0.000000000 */  0.006666666666667, 0.040000000000000, /* 0.000000000 */  0.013333333333333,
    0.016666666666667, /* 0.000000000 */  0.006666666666667, /* 0.000000000 */  0.006666666666667, /* 0.000000000 */  0.040000000000000, 0.013333333333333,
    0.050000000000000, 0.016666666666667, 0.016666666666667, 0.013333333333333, 0.016666666666667, 0.013333333333333, 0.013333333333333, 0.110000000000000};

  auto tHostRowMap = Kokkos::create_mirror_view(jacobian->rowMap());
  auto tHostColumnIndices = Kokkos::create_mirror_view(jacobian->columnIndices());
  auto tHostValues = Kokkos::create_mirror_view(jacobian->entries());
  Kokkos::deep_copy(tHostRowMap, jacobian->rowMap());
  Kokkos::deep_copy(tHostColumnIndices, jacobian->columnIndices());
  Kokkos::deep_copy(tHostValues, jacobian->entries());

  TEST_COMPARE_ARRAYS(tHostRowMap, tRowMapA);
  TEST_COMPARE_ARRAYS(tHostColumnIndices, tColIndices);
  TEST_COMPARE_FLOATING_ARRAYS(tHostValues, tValuesA, 1e-13);
}
