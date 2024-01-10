/*
 *  StructuralDynamicsTest.cpp
 *  
 *   Created on: May 15, 2018
 **/

#include <iostream>
#include <fstream>

#include "util/PlatoTestHelpers.hpp"
 
#include "Mechanics.hpp"
#include "WorksetBase.hpp"
#include "elliptic/VectorFunction.hpp"
#include "ImplicitFunctors.hpp"
#include "StructuralDynamics.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

    // create material model
    //
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                           \n"
      "  <ParameterList name='Spatial Model'>                                         \n"
      "    <ParameterList name='Domains'>                                             \n"
      "      <ParameterList name='Design Volume'>                                     \n"
      "        <Parameter name='Element Block' type='string' value='body'/>           \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>   \n"
      "      </ParameterList>                                                         \n"
      "    </ParameterList>                                                           \n"
      "  </ParameterList>                                                             \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elastostatics'/>       \n"
      "  <Parameter name='Objective' type='string' value='My Internal Elastic Energy'/> \n"
      "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                    \n"
      "  <ParameterList name='Elastostatics'>                                         \n"
      "    <ParameterList name='Penalty Function'>                                    \n"
      "      <Parameter name='Exponent' type='double' value='1.0'/>                   \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                      \n"
      "    </ParameterList>                                                           \n"
      "  </ParameterList>                                                             \n"
      "  <ParameterList name='My Internal Elastic Energy'>                            \n"
      "    <Parameter name='Type' type='string' value='Scalar Function'/>             \n"
      "    <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
      "  </ParameterList>                                                             \n"
      "  <ParameterList name='Material Models'>                                       \n"
      "    <ParameterList name='Unobtainium'>                                         \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                          \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>           \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>         \n"
      "      </ParameterList>                                                         \n"
      "    </ParameterList>                                                           \n"
      "  </ParameterList>                                                             \n"
      "</ParameterList>                                                               \n"
    );
 

namespace PlatoUnitTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ComputeStateWorkset)
{
    // ****** TEST STATE WORKSET TOOLS ****** //
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::Residual;
    using JacobianU = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::Jacobian;
    using JacobianX = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::GradientX;
    using JacobianZ = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTOSTATICS RESIDUAL

    Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Plato::Elliptic::VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(tSpatialModel, tDataMap);

    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::Elliptic::ElastostaticResidual<ResidualT, Plato::MSIMP>>
        (tOnlyDomain, tDataMap, *tParamList, tParamList->sublist("Elastostatics"));
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::Elliptic::ElastostaticResidual<JacobianU, Plato::MSIMP>>
        (tOnlyDomain, tDataMap, *tParamList, tParamList->sublist("Elastostatics"));
    tElastostatics.setEvaluator(tResidual, tJacobianState, tOnlyDomain.getDomainName());

    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh->NumElements();
    Plato::OrdinalType tNumVertices = tMesh->NumNodes();
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    
    // ALLOCATE STATES VECTOR FOR ELASTODYNAMICS EXAMPLE
    Plato::ScalarVector tStateReal("Real States", tTotalNumDofs);
    Plato::ScalarVector tStateImag("Imag States", tTotalNumDofs);
    auto tHostStateReal = Kokkos::create_mirror(tStateReal);
    auto tHostStateImag = Kokkos::create_mirror(tStateImag);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostStateReal(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostStateImag(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tStateReal, tHostStateReal);
    Kokkos::deep_copy(tStateImag, tHostStateImag);
    
    // ALLOCATE STATE WORKSET FOR ELASTODYNAMICS EXAMPLE
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + static_cast<Plato::OrdinalType>(1);
    Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tStateRealWS("Real States Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tStateReal, tStateRealWS);
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tStateImagWS("Imag States Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tStateImag, tStateImagWS);

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTODYNAMICS RESIDUAL 
    Plato::Elliptic::VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(tSpatialModel, tDataMap);
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
        (tOnlyDomain, tDataMap);
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
        (tOnlyDomain, tDataMap);
    tElastodynamics.setEvaluator(tResidualSD, tJacobianStateSD, tOnlyDomain.getDomainName());

    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    tNumDofsPerCell = static_cast<Plato::OrdinalType>(2) * tSpaceDim * tNumNodesPerCell;
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyRealIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyRealIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        Plato::OrdinalType tMyImagIndex = (tIndex % tSpaceDim) + tSpaceDim
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyImagIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarMultiVectorT<SD_ResidualT::StateScalarType> tComplexStatesWS("ComplexStatesWS", tNumCells, tNumDofsPerCell);
    tElastodynamics.worksetState(tComplexStates, tComplexStatesWS);

    // TEST WORKSET OUTPUTS
    auto tHostStateRealWS = Kokkos::create_mirror(tStateRealWS);
    Kokkos::deep_copy(tHostStateRealWS, tStateRealWS);
    auto tHostStateImagWS = Kokkos::create_mirror(tStateImagWS);
    Kokkos::deep_copy(tHostStateImagWS, tStateImagWS);
    auto tHostComplexStatesWS = Kokkos::create_mirror(tComplexStatesWS);
    Kokkos::deep_copy(tHostComplexStatesWS, tComplexStatesWS);

    const Plato::Scalar tTolerance = 1e-6;
    const Plato::OrdinalType tNumRealDofsPerCell = tSpaceDim * tNumNodesPerCell;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumRealDofsPerCell; tDofIndex++)
        {
            Plato::OrdinalType tMyRealIndex = (tDofIndex % tSpaceDim)
                + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
            TEST_FLOATING_EQUALITY(tHostComplexStatesWS(tCellIndex, tMyRealIndex), tHostStateRealWS(tCellIndex, tDofIndex), tTolerance);
            Plato::OrdinalType tMyImagIndex = (tDofIndex % tSpaceDim) + tSpaceDim
                + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
            TEST_FLOATING_EQUALITY(tHostComplexStatesWS(tCellIndex, tMyImagIndex), tHostStateImagWS(tCellIndex, tDofIndex), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CompareLinearStrainsToComplexStrains)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::Residual;
    using JacobianU = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::Jacobian;
    using JacobianX = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::GradientX;
    using JacobianZ = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;
    
    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION

    Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::DataMap tDataMap;
    Plato::Elliptic::VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(tSpatialModel, tDataMap);

    // ALLOCATE ELASTOSTATICS RESIDUAL
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::Elliptic::ElastostaticResidual<ResidualT, Plato::MSIMP>>
        (tOnlyDomain, tDataMap, *tParamList, tParamList->sublist("Elastostatics"));
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::Elliptic::ElastostaticResidual<JacobianU, Plato::MSIMP>>
        (tOnlyDomain, tDataMap, *tParamList, tParamList->sublist("Elastostatics"));
    tElastostatics.setEvaluator(tResidual, tJacobianState, tOnlyDomain.getDomainName());

    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh->NumElements();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(6));
    Plato::OrdinalType tNumVertices = tMesh->NumNodes();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(8));
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(24));

    // ALLOCATE STATES VECTOR FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tRealStates("Real LinearStates", tTotalNumDofs);
    auto tHostRealStates = Kokkos::create_mirror(tRealStates);
    Plato::ScalarVector tImagStates("Imag LinearStates", tTotalNumDofs);
    auto tHostImagStates = Kokkos::create_mirror(tImagStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostRealStates(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostImagStates(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }    
    Kokkos::deep_copy(tRealStates, tHostRealStates);
    Kokkos::deep_copy(tImagStates, tHostImagStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + static_cast<Plato::OrdinalType>(1);
    Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tRealStatesWS("Real LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tRealStates, tRealStatesWS);
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tImagStatesWS("Imag LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tImagStates, tImagStatesWS);

    // ALLOCATE COMMON DATA STRUCTURES FOR ELASTOSTATICS AND ELASTODYNAMICS EXAMPLES
    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::ScalarVectorT<ResidualT::ConfigScalarType> tCellVolume("Cell Volume", tNumCells);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tGradient("Gradient", tNumCells, tNumNodesPerCell, tSpaceDim);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfigWS("Configuration Workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tElastostatics.worksetConfig(tConfigWS);

    // COMPUTE LINEAR STRAINS 
    Plato::Strain<tSpaceDim> tComputeLinearStrain;
    const Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarMultiVectorT<StrainT> tRealLinearStrain("RealLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<StrainT> tImagLinearStrain("ImagLinearStrain", tNumCells, tNumVoigtTerms);
    Kokkos::parallel_for("UnitTest::LinearStrains", Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        // compute strain
        tComputeLinearStrain(aCellOrdinal, tRealLinearStrain, tRealStatesWS, tGradient);
        tComputeLinearStrain(aCellOrdinal, tImagLinearStrain, tImagStatesWS, tGradient);
    });

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;
    
    // ALLOCATE ELASTODYNAMICS VECTOR FUNCTION
    Plato::Elliptic::VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(tSpatialModel, tDataMap);
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
        (tOnlyDomain, tDataMap);
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
        (tOnlyDomain, tDataMap);
    tElastodynamics.setEvaluator(tResidualSD, tJacobianStateSD, tOnlyDomain.getDomainName());

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(48));
    tNumDofsPerCell = static_cast<Plato::OrdinalType>(2) * tSpaceDim * tNumNodesPerCell;
    TEST_EQUALITY(tNumDofsPerCell, static_cast<Plato::OrdinalType>(24));
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tMyIndex = (tIndex % tSpaceDim) + tSpaceDim + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);
    
    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarMultiVectorT<SD_ResidualT::StateScalarType> tComplexStatesWS("ComplexStatesWS", tNumCells, tNumDofsPerCell);
    tElastodynamics.worksetState(tComplexStates, tComplexStatesWS);
    
    // COMPUTE COMPLEX STRAINS
    const Plato::OrdinalType tCOMPLEX_SPACE_DIM = 2;
    Plato::ComplexStrain<tSpaceDim, tNumDofsPerNode> tComputeComplexStrain;
    Plato::ScalarArray3DT<StrainT> tComplexStrain("ComplexStrain", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);

    Kokkos::parallel_for("UnitTest::ComplexStrain", Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        tCellVolume(aCellOrdinal) = 0.0;
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        // compute strain
        tComputeComplexStrain(aCellOrdinal, tComplexStatesWS, tGradient, tComplexStrain);
    });

    // TEST OUTPUTS: LINEAR STRAINS AND COMPLEX STRAINS SHOULD BE EQUAL
    auto tHostRealLinearStrain = Kokkos::create_mirror(tRealLinearStrain);
    Kokkos::deep_copy(tHostRealLinearStrain, tRealLinearStrain);
    auto tHostImagLinearStrain = Kokkos::create_mirror(tImagLinearStrain);
    Kokkos::deep_copy(tHostImagLinearStrain, tImagLinearStrain);
    auto tHostComplexStrain = Kokkos::create_mirror(tComplexStrain);
    Kokkos::deep_copy(tHostComplexStrain, tComplexStrain);
    
    const Plato::Scalar tTolerance = 1e-6;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tVoigtIndex = 0; tVoigtIndex < tNumVoigtTerms; tVoigtIndex++)
        {
            Plato::OrdinalType tCOMPLEX_SPACE_INDEX = 0;
            TEST_FLOATING_EQUALITY(tHostComplexStrain(tCellIndex, tCOMPLEX_SPACE_INDEX, tVoigtIndex), tHostRealLinearStrain(tCellIndex, tVoigtIndex), tTolerance);
            tCOMPLEX_SPACE_INDEX = 1;
            TEST_FLOATING_EQUALITY(tHostComplexStrain(tCellIndex, tCOMPLEX_SPACE_INDEX, tVoigtIndex), tHostImagLinearStrain(tCellIndex, tVoigtIndex), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CompareLinearStressToComplexStress)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;

    using SimplexPhysics = typename Plato::SimplexMechanics<tSpaceDim>;

    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::Residual;
    using JacobianU = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::Jacobian;
    using JacobianX = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::GradientX;
    using JacobianZ = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTOSTATICS RESIDUAL

    Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Plato::Elliptic::VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(tSpatialModel, tDataMap);

    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::Elliptic::ElastostaticResidual<ResidualT, Plato::MSIMP>>
        (tOnlyDomain, tDataMap, *tParamList, tParamList->sublist("Elastostatics"));
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::Elliptic::ElastostaticResidual<JacobianU, Plato::MSIMP>>
        (tOnlyDomain, tDataMap, *tParamList, tParamList->sublist("Elastostatics"));
    tElastostatics.setEvaluator(tResidual, tJacobianState, tOnlyDomain.getDomainName());
    
    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh->NumElements();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(6));
    Plato::OrdinalType tNumVertices = tMesh->NumNodes();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(8));
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(24));

    // ALLOCATE STATES VECTOR FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tRealStates("Real LinearStates", tTotalNumDofs);
    auto tHostRealStates = Kokkos::create_mirror(tRealStates);
    Plato::ScalarVector tImagStates("Imag LinearStates", tTotalNumDofs);
    auto tHostImagStates = Kokkos::create_mirror(tImagStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostRealStates(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostImagStates(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tRealStates, tHostRealStates);
    Kokkos::deep_copy(tImagStates, tHostImagStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + static_cast<Plato::OrdinalType>(1);
    Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tRealStatesWS("Real LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tRealStates, tRealStatesWS);
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tImagStatesWS("Imag LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tImagStates, tImagStatesWS);

    // ALLOCATE COMMON DATA STRUCTURES FOR ELASTOSTATICS AND ELASTODYNAMICS EXAMPLES
    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::ScalarVectorT<ResidualT::ConfigScalarType> tCellVolume("Cell Volume", tNumCells);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tGradient("Gradient", tNumCells, tNumNodesPerCell, tSpaceDim);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfigWS("Configuration Workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tElastostatics.worksetConfig(tConfigWS);

    // COMPUTE LINEAR STRESS
    Plato::Scalar tYoungModulus = 1.0;
    Plato::Scalar tPoissonRatio = 0.3;
    Plato::Strain<tSpaceDim> tComputeLinearStrain;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMaterialModel(tYoungModulus, tPoissonRatio);
    auto tStiffnessMatrix = tMaterialModel.getStiffnessMatrix();
    Plato::LinearStress<Plato::ResidualTypes<SimplexPhysics>,
                        SimplexPhysics> tComputeLinearStress(tStiffnessMatrix);

    const Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarMultiVectorT<StrainT> tRealLinearStrain("RealLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tRealLinearStress("RealLinearStress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<StrainT> tImagLinearStrain("ImagLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tImagLinearStress("ImagLinearStress", tNumCells, tNumVoigtTerms);
    Kokkos::parallel_for("UnitTest::LinearStress", Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);

        // compute strain
        tComputeLinearStrain(aCellOrdinal, tRealLinearStrain, tRealStatesWS, tGradient);
        tComputeLinearStrain(aCellOrdinal, tImagLinearStrain, tImagStatesWS, tGradient);

        // compute stress
        tComputeLinearStress(aCellOrdinal, tRealLinearStress, tRealLinearStrain);
        tComputeLinearStress(aCellOrdinal, tImagLinearStress, tImagLinearStrain);
    });

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTODYNAMICS VECTOR FUNCTION
    Plato::Elliptic::VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(tSpatialModel, tDataMap);
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
        (tOnlyDomain, tDataMap);
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
        (tOnlyDomain, tDataMap);
    tElastodynamics.setEvaluator(tResidualSD, tJacobianStateSD, tOnlyDomain.getDomainName());

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(48));
    tNumDofsPerCell = static_cast<Plato::OrdinalType>(2) * tSpaceDim * tNumNodesPerCell;
    TEST_EQUALITY(tNumDofsPerCell, static_cast<Plato::OrdinalType>(24));
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tMyIndex = (tIndex % tSpaceDim) + tSpaceDim + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarMultiVectorT<SD_ResidualT::StateScalarType> tComplexStatesWS("ComplexStatesWS", tNumCells, tNumDofsPerCell);
    tElastodynamics.worksetState(tComplexStates, tComplexStatesWS);

    // COMPUTE COMPLEX STRESSES
    const Plato::OrdinalType tCOMPLEX_SPACE_DIM = 2;
    Plato::ComplexStrain<tSpaceDim, tNumDofsPerNode> tComputeComplexStrain;
    Plato::ComplexLinearStress<tSpaceDim, tNumVoigtTerms> tComputeComplexStress(tStiffnessMatrix);
    Plato::ScalarArray3DT<StrainT>
        tComplexStrain("ComplexStrain", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);
    Plato::ScalarArray3DT<ResidualT::ResultScalarType>
        tComplexStress("ComplexStress", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);

    Kokkos::parallel_for("UnitTest::ComplexStress", Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        tCellVolume(aCellOrdinal) = 0.0;
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        // compute strain
        tComputeComplexStrain(aCellOrdinal, tComplexStatesWS, tGradient, tComplexStrain);
        // compute stress
        tComputeComplexStress(aCellOrdinal, tComplexStrain, tComplexStress);
    });

    // TEST OUTPUTS: LINEAR STRESSES AND COMPLEX STRESSES SHOULD BE EQUAL
    auto tHostRealLinearStress = Kokkos::create_mirror(tRealLinearStress);
    Kokkos::deep_copy(tHostRealLinearStress, tRealLinearStress);
    auto tHostImagLinearStress = Kokkos::create_mirror(tImagLinearStress);
    Kokkos::deep_copy(tHostImagLinearStress, tImagLinearStress);
    auto tHostComplexStress = Kokkos::create_mirror(tComplexStress);
    Kokkos::deep_copy(tHostComplexStress, tComplexStress);

    const Plato::Scalar tTolerance = 1e-6;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tVoigtIndex = 0; tVoigtIndex < tNumVoigtTerms; tVoigtIndex++)
        {
            Plato::OrdinalType tCOMPLEX_SPACE_INDEX = 0;
            TEST_FLOATING_EQUALITY(tHostComplexStress(tCellIndex, tCOMPLEX_SPACE_INDEX, tVoigtIndex), tHostRealLinearStress(tCellIndex, tVoigtIndex), tTolerance);
            tCOMPLEX_SPACE_INDEX = 1;
            TEST_FLOATING_EQUALITY(tHostComplexStress(tCellIndex, tCOMPLEX_SPACE_INDEX, tVoigtIndex), tHostImagLinearStress(tCellIndex, tVoigtIndex), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CompareLinearElasticForcesToComplexElasticForces)
{
    // BUILD OMEGA_H MESH
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tMeshWidth = 1;

    using SimplexPhysics = typename Plato::SimplexMechanics<tSpaceDim>;

    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", tMeshWidth);

    // ******************** SET ELASTOSTATICS' EVALUATION TYPES FOR UNIT TEST ********************
    using ResidualT = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::Residual;
    using JacobianU = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::Jacobian;
    using JacobianX = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::GradientX;
    using JacobianZ = typename Plato::Evaluation<typename Plato::Mechanics<tSpaceDim>::SimplexT>::GradientZ;
    using StrainT = typename Plato::fad_type_t<Plato::Mechanics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTOSTATICS RESIDUAL

    Plato::SpatialModel tSpatialModel(tMesh, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    // ALLOCATE ELASTOSTATICS VECTOR FUNCTION
    Plato::DataMap tDataMap;
    Plato::Elliptic::VectorFunction<Plato::Mechanics<tSpaceDim>> tElastostatics(tSpatialModel, tDataMap);
    
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<ResidualT>> tResidual;
    tResidual = std::make_shared<Plato::Elliptic::ElastostaticResidual<ResidualT, Plato::MSIMP>>
        (tOnlyDomain, tDataMap, *tParamList, tParamList->sublist("Elastostatics"));
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<JacobianU>> tJacobianState;
    tJacobianState = std::make_shared<Plato::Elliptic::ElastostaticResidual<JacobianU, Plato::MSIMP>>
        (tOnlyDomain, tDataMap, *tParamList, tParamList->sublist("Elastostatics"));
    tElastostatics.setEvaluator(tResidual, tJacobianState, tOnlyDomain.getDomainName());

    // SET PROBLEM-RELATED DIMENSIONS
    Plato::OrdinalType tNumCells = tMesh->NumElements();
    TEST_EQUALITY(tNumCells, static_cast<Plato::OrdinalType>(6));
    Plato::OrdinalType tNumVertices = tMesh->NumNodes();
    TEST_EQUALITY(tNumVertices, static_cast<Plato::OrdinalType>(8));
    Plato::OrdinalType tTotalNumDofs = tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(24));

    // ALLOCATE STATES VECTOR FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarVector tRealStates("Real LinearStates", tTotalNumDofs);
    auto tHostRealStates = Kokkos::create_mirror(tRealStates);
    Plato::ScalarVector tImagStates("Imag LinearStates", tTotalNumDofs);
    auto tHostImagStates = Kokkos::create_mirror(tImagStates);
    for(Plato::OrdinalType tIndex = 0; tIndex < tTotalNumDofs; tIndex++)
    {
        tHostRealStates(tIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tHostImagStates(tIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tRealStates, tHostRealStates);
    Kokkos::deep_copy(tImagStates, tHostImagStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::OrdinalType tNumNodesPerCell = tSpaceDim + static_cast<Plato::OrdinalType>(1);
    Plato::OrdinalType tNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tRealStatesWS("Real LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tRealStates, tRealStatesWS);
    Plato::ScalarMultiVectorT<ResidualT::StateScalarType> tImagStatesWS("Imag LinearStates Workset", tNumCells, tNumDofsPerCell);
    tElastostatics.worksetState(tImagStates, tImagStatesWS);

    // ALLOCATE COMMON DATA STRUCTURES FOR ELASTOSTATICS AND ELASTODYNAMICS EXAMPLES
    Plato::ComputeGradientWorkset<tSpaceDim> tComputeGradient;
    Plato::LinearTetCubRuleDegreeOne<tSpaceDim> tCubatureRule;
    Plato::ScalarVectorT<ResidualT::ConfigScalarType> tCellVolume("Cell Volume", tNumCells);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tGradient("Gradient", tNumCells, tNumNodesPerCell, tSpaceDim);
    Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfigWS("Configuration Workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tElastostatics.worksetConfig(tConfigWS);

    // COMPUTE LINEAR ELASTIC FORCES
    Plato::Scalar tYoungModulus = 1.0;
    Plato::Scalar tPoissonRatio = 0.3;
    Plato::Strain<tSpaceDim> tComputeLinearStrain;
    Plato::StressDivergence<tSpaceDim> tComputeElasticForces;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMaterialModel(tYoungModulus, tPoissonRatio);
    auto tStiffnessMatrix = tMaterialModel.getStiffnessMatrix();
    Plato::LinearStress<Plato::ResidualTypes<SimplexPhysics>,
                        SimplexPhysics> tComputeLinearStress(tStiffnessMatrix);

    const Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarMultiVectorT<StrainT> tRealLinearStrain("RealLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tRealLinearStress("RealLinearStress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tRealElasticForces("RealElasticForces", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVectorT<StrainT> tImagLinearStrain("ImagLinearStrain", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tImagLinearStress("ImagLinearStress", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType> tImagElasticForces("ImagElasticForces", tNumCells, tNumDofsPerCell);

    auto tQuadratureWeight = tCubatureRule.getCubWeight();
    Kokkos::parallel_for("UnitTest::ElasticForces", Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        tCellVolume(aCellOrdinal) = 0.0;
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;
        // compute strain
        tComputeLinearStrain(aCellOrdinal, tRealLinearStrain, tRealStatesWS, tGradient);
        tComputeLinearStrain(aCellOrdinal, tImagLinearStrain, tImagStatesWS, tGradient);
        // compute stress
        tComputeLinearStress(aCellOrdinal, tRealLinearStress, tRealLinearStrain);
        tComputeLinearStress(aCellOrdinal, tImagLinearStress, tImagLinearStrain);
        // compute elastic forces
        tComputeElasticForces(aCellOrdinal, tRealElasticForces, tRealLinearStress, tGradient, tCellVolume);
        tComputeElasticForces(aCellOrdinal, tImagElasticForces, tImagLinearStress, tGradient, tCellVolume);
    });

    // ******************** SET ELASTODYNAMICS' EVALUATION TYPES FOR UNIT TEST ********************
    using SD_ResidualT = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Residual;
    using SD_JacobianU = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::Jacobian;
    using SD_JacobianX = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::GradientX;
    using SD_JacobianZ = typename Plato::Evaluation<typename Plato::StructuralDynamics<tSpaceDim>::SimplexT>::GradientZ;
    using SD_StrainT = typename
        Plato::fad_type_t<Plato::StructuralDynamics<tSpaceDim>, ResidualT::StateScalarType, ResidualT::ConfigScalarType>;

    // ALLOCATE ELASTODYNAMICS VECTOR FUNCTION
    Plato::Elliptic::VectorFunction<Plato::StructuralDynamics<tSpaceDim>> tElastodynamics(tSpatialModel, tDataMap);
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<SD_ResidualT>> tResidualSD;
    tResidualSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_ResidualT, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
        (tOnlyDomain, tDataMap);
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<SD_JacobianU>> tJacobianStateSD;
    tJacobianStateSD = std::make_shared<Plato::StructuralDynamicsResidual<SD_JacobianU, Plato::MSIMP, Plato::HyperbolicTangentProjection>>
        (tOnlyDomain, tDataMap);
    tElastodynamics.setEvaluator(tResidualSD, tJacobianStateSD, tOnlyDomain.getDomainName());

    // ALLOCATE STATE VECTOR FOR ELASTODYNAMICS EXAMPLE
    tTotalNumDofs = static_cast<Plato::OrdinalType>(2) * tNumVertices * tSpaceDim;
    TEST_EQUALITY(tTotalNumDofs, static_cast<Plato::OrdinalType>(48));
    tNumDofsPerCell = static_cast<Plato::OrdinalType>(2) * tSpaceDim * tNumNodesPerCell;
    TEST_EQUALITY(tNumDofsPerCell, static_cast<Plato::OrdinalType>(24));
    Plato::ScalarVector tComplexStates("ComplexStates", tTotalNumDofs);

    auto tHostComplexStates = Kokkos::create_mirror(tComplexStates);
    const Plato::OrdinalType tNumRealDofs = tNumVertices * tSpaceDim;
    const Plato::OrdinalType tNumDofsPerNode = static_cast<Plato::OrdinalType>(2) * tSpaceDim;
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumRealDofs; tIndex++)
    {
        Plato::OrdinalType tMyIndex = (tIndex % tSpaceDim)
            + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(1e-3) * static_cast<Plato::Scalar>(tIndex);
        tMyIndex = (tIndex % tSpaceDim) + tSpaceDim + (static_cast<Plato::OrdinalType>(tIndex/tSpaceDim) * tNumDofsPerNode);
        tHostComplexStates(tMyIndex) = static_cast<Plato::Scalar>(2e-3) * static_cast<Plato::Scalar>(tIndex);
    }
    Kokkos::deep_copy(tComplexStates, tHostComplexStates);

    // ALLOCATE STATE WORKSET FOR ELASTOSTATICS EXAMPLE
    Plato::ScalarMultiVectorT<SD_ResidualT::StateScalarType> tComplexStatesWS("ComplexStatesWS", tNumCells, tNumDofsPerCell);
    tElastodynamics.worksetState(tComplexStates, tComplexStatesWS);

    // COMPUTE COMPLEX ELASTIC FORCES
    const Plato::OrdinalType tCOMPLEX_SPACE_DIM = 2;
    Plato::ComplexStrain<tSpaceDim, tNumDofsPerNode> tComputeComplexStrain;
    Plato::ComplexLinearStress<tSpaceDim, tNumVoigtTerms> tComputeComplexStress(tStiffnessMatrix);
    Plato::ComplexStressDivergence<tSpaceDim, tNumDofsPerNode> tComputeComplexElasticForces;
    Plato::ScalarArray3DT<StrainT>
        tComplexStrain("ComplexStrain", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);
    Plato::ScalarArray3DT<ResidualT::ResultScalarType>
        tComplexStress("ComplexStress", tNumCells, tCOMPLEX_SPACE_DIM, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<ResidualT::ResultScalarType>
        tComplexElasticForces("ComplexElasticForces", tNumCells, tNumDofsPerCell);

    Kokkos::parallel_for("UnitTest::ComplexElasticForces", Kokkos::RangePolicy<>(0,tNumCells), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        tCellVolume(aCellOrdinal) = 0.0;
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;
        // compute strain
        tComputeComplexStrain(aCellOrdinal, tComplexStatesWS, tGradient, tComplexStrain);
        // compute stress
        tComputeComplexStress(aCellOrdinal, tComplexStrain, tComplexStress);
        // compute elastic forces
        tComputeComplexElasticForces(aCellOrdinal, tCellVolume, tGradient, tComplexStress, tComplexElasticForces);
    });

    // TEST OUTPUTS: LINEAR AND COMPLEX ELASTIC FORCES SHOULD BE EQUAL
    auto tHostRealElasticForces = Kokkos::create_mirror(tRealElasticForces);
    Kokkos::deep_copy(tHostRealElasticForces, tRealElasticForces);
    auto tHostImagElasticForces = Kokkos::create_mirror(tImagElasticForces);
    Kokkos::deep_copy(tHostImagElasticForces, tImagElasticForces);
    auto tHostComplexElasticForces = Kokkos::create_mirror(tComplexElasticForces);
    Kokkos::deep_copy(tHostComplexElasticForces, tComplexElasticForces);

    const Plato::Scalar tTolerance = 1e-6;
    const Plato::OrdinalType tRealNumDofsPerCell = tSpaceDim * tNumNodesPerCell;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tRealNumDofsPerCell; tDofIndex++)
        {
            Plato::OrdinalType tMyIndex = (tDofIndex % tSpaceDim)
                    + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
            TEST_FLOATING_EQUALITY(tHostComplexElasticForces(tCellIndex, tMyIndex), tHostRealElasticForces(tCellIndex, tDofIndex), tTolerance);
            tMyIndex = (tDofIndex % tSpaceDim) + tSpaceDim
                    + (static_cast<Plato::OrdinalType>(tDofIndex/tSpaceDim) * tNumDofsPerNode);
            TEST_FLOATING_EQUALITY(tHostComplexElasticForces(tCellIndex, tMyIndex), tHostImagElasticForces(tCellIndex, tDofIndex), tTolerance);
        }
    }
}

} // namespace PlatoUnitTests
