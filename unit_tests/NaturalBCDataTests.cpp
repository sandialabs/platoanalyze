#include "Hex8.hpp"
#include "NaturalBCData.hpp"
#include "util/PlatoTestHelpers.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>

namespace
{
/// The purpose of this function to emulate using scalarBoundaryDataAtIndex in a kokkos
/// parallelized section for testing.
template<unsigned int NumIndices>
Plato::ScalarVector boundaryDataAtIndices(const Plato::NaturalBCScalarData& aData, 
    const Plato::Array<NumIndices, unsigned int>& aIndices)
{
    Plato::ScalarVector tOut("test bc data", NumIndices);
    Kokkos::parallel_for(aIndices.size(),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aIndex)
    {
        tOut(aIndex) = Plato::scalarBoundaryDataAtIndex(aData, aIndices[aIndex]);
    });
    return tOut;
}

/// The purpose of this function to emulate using scalarBoundaryDataAtIndex in a kokkos
/// parallelized section for testing.
template<Plato::OrdinalType NumDofs, unsigned int NumIndices>
Plato::ScalarVector boundaryDataAtIndices(const Plato::NaturalBCVectorData<NumDofs>& aData, 
    const Plato::Array<NumIndices, unsigned int>& aIndices,
    const Plato::OrdinalType aDofIndex)
{
    Plato::ScalarVector tOut("test bc data", NumIndices);
    Kokkos::parallel_for(aIndices.size(),
    KOKKOS_LAMBDA(const Plato::OrdinalType & aIndex)
    {
        tOut(aIndex) = Plato::vectorBoundaryDataAtIndex<NumDofs>(aData, aIndices[aIndex])[aDofIndex];
    });
    return tOut;
}

}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, UniformPressure)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Test Boundary Condition'>\n"
        "  <Parameter name='Type' type='string' value='Uniform pressure'/>\n"
        "  <Parameter name='Value' type='double' value='1'/>\n"
        "  <Parameter name='Sides' type='string' value='x+'/>\n"
        "</ParameterList>\n"
    );

    constexpr int kNumDofs = 3;
    std::unique_ptr<Plato::NaturalBCData<kNumDofs>> tBCData = Plato::makeNaturalBCData<kNumDofs>(*tInputs);

    TEST_ASSERT(tBCData != nullptr);
    using ExpectedType = Plato::UniformNaturalBCData<kNumDofs, Plato::BCDataType::kScalar>;
    TEST_ASSERT(dynamic_cast<ExpectedType*>(tBCData.get()) != nullptr);
    TEST_ASSERT(tBCData->getDataType() == Plato::BCDataType::kScalar);

    const auto tScalarBoundaryData = tBCData->getScalarData();

    constexpr unsigned int kNumIndices = 3;
    Plato::ScalarVector tResult = boundaryDataAtIndices<kNumIndices>(tScalarBoundaryData, {0, 2, 42});
    auto tHostMirror = Kokkos::create_mirror_view(tResult);
    Kokkos::deep_copy(tHostMirror, tResult);
    TEST_EQUALITY(tHostMirror(0), 1.0);
    TEST_EQUALITY(tHostMirror(1), 1.0);
    TEST_EQUALITY(tHostMirror(2), 1.0);
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, Uniform)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Test Boundary Condition'>\n"
        "  <Parameter name='Type' type='string' value='Uniform'/>\n"
        "  <Parameter name='Values' type='Array(double)' value='{1, 2, 3}'/>\n"
        "  <Parameter name='Sides' type='string' value='z-'/>\n"
        "</ParameterList>\n"
    );

    constexpr int kNumDofs = 3;
    std::unique_ptr<Plato::NaturalBCData<kNumDofs>> tBCData = Plato::makeNaturalBCData<kNumDofs>(*tInputs);

    TEST_ASSERT(tBCData != nullptr);
    using ExpectedType = Plato::UniformNaturalBCData<kNumDofs, Plato::BCDataType::kVector>;
    TEST_ASSERT(dynamic_cast<ExpectedType*>(tBCData.get()) !=  nullptr);
    TEST_ASSERT(tBCData->getDataType() == Plato::BCDataType::kVector);

    const auto tVectorBoundaryData = tBCData->getVectorData();
    TEST_ASSERT(tVectorBoundaryData.mValue.size() == kNumDofs);
    auto tHostMirror = Kokkos::create_mirror_view(tVectorBoundaryData.mValue);
    Kokkos::deep_copy(tHostMirror, tVectorBoundaryData.mValue);
    TEST_EQUALITY(tHostMirror.extent(0), 1);
    for(Plato::OrdinalType i = 0; i < kNumDofs; ++i)
    {
        TEST_EQUALITY(tHostMirror(0, i), static_cast<double>(i + 1));
    }
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, TimeVarying)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Test Boundary Condition'>\n"
        "  <Parameter name='Type' type='string' value='Uniform'/>\n"
        "  <Parameter name='Values' type='Array(string)' value='{t, 2 * t}'/>\n"
        "  <Parameter name='Sides' type='string' value='z-'/>\n"
        "</ParameterList>\n"
    );

    constexpr int kNumDofs = 2;
    std::unique_ptr<Plato::NaturalBCData<kNumDofs>> tBCData = Plato::makeNaturalBCData<kNumDofs>(*tInputs);

    TEST_ASSERT(tBCData != nullptr);
    using ExpectedType = Plato::TimeVaryingNaturalBCData<kNumDofs, Plato::BCDataType::kVector>;
    TEST_ASSERT(dynamic_cast<ExpectedType*>(tBCData.get()) != nullptr);
    TEST_ASSERT(tBCData->getDataType() == Plato::BCDataType::kVector);

    for(int t = 0; t < 4; ++t)
    {
        const auto tVectorBoundaryData = tBCData->getVectorData(t);
        auto tHostMirror = Kokkos::create_mirror_view(tVectorBoundaryData.mValue);
        Kokkos::deep_copy(tHostMirror, tVectorBoundaryData.mValue);
        TEST_EQUALITY(tHostMirror.extent(0), 1);
        for(Plato::OrdinalType iDof = 0; iDof < kNumDofs; ++iDof)
        {
            TEST_EQUALITY(tHostMirror(0, iDof), static_cast<double>((iDof + 1) * t));
        }
    }
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, TimeVaryingPressure)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Test Boundary Condition'>\n"
        "  <Parameter name='Type' type='string' value='Uniform pressure'/>\n"
        "  <Parameter name='Value' type='string' value='2 * t'/>\n"
        "  <Parameter name='Sides' type='string' value='z-'/>\n"
        "</ParameterList>\n"
    );

    constexpr int kNumDofs = 2;
    std::unique_ptr<Plato::NaturalBCData<kNumDofs>> tBCData = Plato::makeNaturalBCData<kNumDofs>(*tInputs);

    TEST_ASSERT(tBCData != nullptr);
    using ExpectedType = Plato::TimeVaryingNaturalBCData<kNumDofs, Plato::BCDataType::kScalar>;
    TEST_ASSERT(dynamic_cast<ExpectedType*>(tBCData.get()) != nullptr);
    TEST_ASSERT(tBCData->getDataType() == Plato::BCDataType::kScalar);

    for(int t = 0; t < 4; ++t)
    {
        const auto tScalarBoundaryData = tBCData->getScalarData(t);
        auto tHostMirror = Kokkos::create_mirror_view(tScalarBoundaryData.mValue);
        Kokkos::deep_copy(tHostMirror, tScalarBoundaryData.mValue);
        TEST_EQUALITY(tHostMirror.size(), 1);
        TEST_EQUALITY(tHostMirror(0), static_cast<double>(2 * t));
    }
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, MeshInputUniform)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Test Boundary Condition'>\n"
        "  <Parameter name='Type' type='string' value='Variable pressure'/>\n"
        "  <Parameter name='Variable' type='string' value='pressure_data'/>\n"
        "  <Parameter name='Sides' type='string' value='z-'/>\n"
        "</ParameterList>\n"
    );

    constexpr int kMeshWidth = 2;
    constexpr auto kMeshName = "test.exo";
    auto tMesh = Plato::TestHelpers::get_box_mesh("TET4", kMeshWidth, kMeshName);
    // Write node data
    {
        Plato::MeshIO tWriter = Plato::MeshIOFactory::create(kMeshName, tMesh, "Write");

        Plato::ScalarVector tDataOut("pressure_data", tMesh->NumNodes());
        Kokkos::deep_copy(tDataOut, 42.0);
        tWriter->AddNodeData("pressure_data", tDataOut);
        constexpr int kStepIndex = 0;
        constexpr double kTime = 0.0;
        tWriter->Write(kStepIndex, kTime);
    }

    constexpr int kNumDofs = 3;
    std::unique_ptr<Plato::NaturalBCData<kNumDofs>> tBCData = Plato::makeNaturalBCData<kNumDofs>(*tInputs);

    TEST_ASSERT(tBCData != nullptr);
    using ExpectedType = Plato::SpatiallyVaryingNaturalBCData<kNumDofs, Plato::BCDataType::kScalar>;
    TEST_ASSERT(dynamic_cast<ExpectedType*>(tBCData.get()) != nullptr);
    TEST_ASSERT(tBCData->getDataType() == Plato::BCDataType::kScalar);

    const auto tBoundaryData = tBCData->getScalarData(tMesh);
    constexpr unsigned int kNumIndices = 3;
    Plato::ScalarVector tResult = boundaryDataAtIndices<kNumIndices>(tBoundaryData, {0, 26, 27});
    auto tHostMirror = Kokkos::create_mirror_view(tResult);
    Kokkos::deep_copy(tHostMirror, tResult);

    TEST_EQUALITY(tBoundaryData.mValue.size(), (kMeshWidth + 1)*(kMeshWidth + 1)*(kMeshWidth + 1));
    TEST_EQUALITY(tHostMirror(0), 42.0);
    TEST_EQUALITY(tHostMirror(1), 42.0);
    TEST_EQUALITY(tHostMirror(2), 42.0);
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, MeshInputVarying)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Test Boundary Condition'>\n"
        "  <Parameter name='Type' type='string' value='Variable pressure'/>\n"
        "  <Parameter name='Variable' type='string' value='pressure_data'/>\n"
        "  <Parameter name='Sides' type='string' value='z-'/>\n"
        "</ParameterList>\n"
    );
    constexpr auto kMeshName = "brick_with_data.exo";
    Plato::Mesh tMesh = Plato::MeshFactory::create(kMeshName);

    constexpr int kNumDofs = 3;
    std::unique_ptr<Plato::NaturalBCData<kNumDofs>> tBCData = Plato::makeNaturalBCData<kNumDofs>(*tInputs);

    TEST_ASSERT(tBCData != nullptr);
    using ExpectedType = Plato::SpatiallyVaryingNaturalBCData<kNumDofs, Plato::BCDataType::kScalar>;
    TEST_ASSERT(dynamic_cast<ExpectedType*>(tBCData.get()) != nullptr);
    TEST_ASSERT(tBCData->getDataType() == Plato::BCDataType::kScalar);

    const auto tBoundaryData = tBCData->getScalarData(tMesh);
    constexpr unsigned int kNumTestNodes = 105;
    TEST_EQUALITY(tBoundaryData.mValue.size(), kNumTestNodes);

    constexpr unsigned int kNumIndices = 3;
    const Plato::Array<kNumIndices, unsigned int> kIndices{0, 26, 27};
    Plato::ScalarVector tResult = boundaryDataAtIndices<kNumIndices>(tBoundaryData, kIndices);
    auto tHostMirror = Kokkos::create_mirror_view(tResult);
    Kokkos::deep_copy(tHostMirror, tResult);

    TEST_EQUALITY(tHostMirror(0), 0.5 * kIndices[0]);
    TEST_EQUALITY(tHostMirror(1), 0.5 * kIndices[1]);
    TEST_EQUALITY(tHostMirror(2), 0.5 * kIndices[2]);
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, MeshInputVaryingLoad)
{
    Teuchos::RCP<Teuchos::ParameterList> tInputs = Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Test Boundary Condition'>\n"
        "  <Parameter name='Type' type='string' value='Variable load'/>\n"
        "  <Parameter name='Variables' type='Array(string)' value='{pressure_data, pressure_data}'/>\n"
        "  <Parameter name='Sides' type='string' value='z-'/>\n"
        "</ParameterList>\n"
    );
    constexpr auto kMeshName = "brick_with_data.exo";
    Plato::Mesh tMesh = Plato::MeshFactory::create(kMeshName);

    constexpr int kNumDofs = 2;
    std::unique_ptr<Plato::NaturalBCData<kNumDofs>> tBCData = Plato::makeNaturalBCData<kNumDofs>(*tInputs);

    TEST_ASSERT(tBCData != nullptr);
    using ExpectedType = Plato::SpatiallyVaryingNaturalBCData<kNumDofs, Plato::BCDataType::kVector>;
    TEST_ASSERT(dynamic_cast<ExpectedType*>(tBCData.get()) != nullptr);
    TEST_ASSERT(tBCData->getDataType() == Plato::BCDataType::kVector);

    const auto tBoundaryData = tBCData->getVectorData(tMesh);
    constexpr unsigned int kNumTestNodes = 105;
    TEST_EQUALITY(tBoundaryData.mValue.size(), kNumDofs * kNumTestNodes);
    TEST_EQUALITY(tBoundaryData.mValue.extent(0), kNumTestNodes);
    TEST_EQUALITY(tBoundaryData.mValue.extent(1), kNumDofs);

    for(Plato::OrdinalType i = 0; i < kNumDofs; ++i)
    {
        constexpr unsigned int kNumIndices = 3;
        const Plato::Array<kNumIndices, unsigned int> kIndices{0, 26, 27};
        Plato::ScalarVector tResult = boundaryDataAtIndices<kNumDofs, kNumIndices>(tBoundaryData, kIndices, i);
        auto tHostMirror = Kokkos::create_mirror_view(tResult);
        Kokkos::deep_copy(tHostMirror, tResult);

        TEST_EQUALITY(tHostMirror(0), 0.5 * kIndices[0]);
        TEST_EQUALITY(tHostMirror(1), 0.5 * kIndices[1]);
        TEST_EQUALITY(tHostMirror(2), 0.5 * kIndices[2]);
    }
}

TEUCHOS_UNIT_TEST(NaturalBCDataTests, MeshInputVaryingElementIteration)
{
     Teuchos::RCP<Teuchos::ParameterList> tInputs = Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Test Boundary Condition'>\n"
        "  <Parameter name='Type' type='string' value='Variable pressure'/>\n"
        "  <Parameter name='Variable' type='string' value='surface_pressure'/>\n"
        "  <Parameter name='Sides' type='string' value='pressure_sideset'/>\n"
        "</ParameterList>\n"
    );
    constexpr auto kMeshName = "nodal_surface_pressure_field.exo";
    Plato::Mesh tMesh = Plato::MeshFactory::create(kMeshName);
   
    constexpr int kNumDofs = 3;
    std::unique_ptr<Plato::NaturalBCData<kNumDofs>> tBCData = Plato::makeNaturalBCData<kNumDofs>(*tInputs);

    TEST_ASSERT(tBCData != nullptr);
    using ExpectedType = Plato::SpatiallyVaryingNaturalBCData<kNumDofs, Plato::BCDataType::kScalar>;
    TEST_ASSERT(dynamic_cast<ExpectedType*>(tBCData.get()) != nullptr);
    TEST_ASSERT(tBCData->getDataType() == Plato::BCDataType::kScalar);

    const auto tBoundaryData = tBCData->getScalarData(tMesh);
    constexpr unsigned int kNumTestNodes = 11 * 11 * 11;
    TEST_EQUALITY(tBoundaryData.mValue.size(), kNumTestNodes);

    constexpr auto kSideSetName = "pressure_sideset";
    using IndexVector = Plato::OrdinalVectorT<const Plato::OrdinalType>;
    const IndexVector tElementOrds = tMesh->GetSideSetElements(kSideSetName);
    const IndexVector tNodeOrds = tMesh->GetSideSetLocalNodes(kSideSetName);
    const IndexVector tFaceOrds = tMesh->GetSideSetFaces(kSideSetName);
    const IndexVector tConnectivity = tMesh->Connectivity();
    const Plato::OrdinalType tNumElements = tElementOrds.size();
    const Plato::OrdinalType tNumFaces = tElementOrds.size();
    const Plato::ScalarVectorT<const Plato::Scalar> tCoordinates = tMesh->Coordinates();

    constexpr Plato::OrdinalType kNumElemsPerEdge = 10;
    TEST_EQUALITY(tElementOrds.size(), kNumElemsPerEdge * kNumElemsPerEdge);

    constexpr Plato::OrdinalType kNumNodesPerSide = (kNumElemsPerEdge + 1) * (kNumElemsPerEdge + 1);

    using ElementType = Plato::Hex8;
    Plato::ScalarVector tSidesetPressures("test bc data", kNumNodesPerSide);
    Plato::ScalarVector tExpectedValue("test expected data", kNumNodesPerSide);

    Kokkos::parallel_for(tElementOrds.size(), 
    KOKKOS_LAMBDA(const Plato::OrdinalType & aSideOrdinal)
    {
        const auto tElementOrdinal = tElementOrds(aSideOrdinal);
        const auto tElemFaceOrdinal = tFaceOrds(aSideOrdinal);

        for(Plato::OrdinalType tNode = 0; tNode < ElementType::mNumNodesPerFace; ++tNode)
        {
            const auto tLocalNodeOrdinal = tNodeOrds(aSideOrdinal*ElementType::mNumNodesPerFace + tNode);
            const auto tGlobalNodeOrdinal = tConnectivity(tElementOrdinal*ElementType::mNumNodesPerCell + tLocalNodeOrdinal);
            const Plato::Scalar tPressure = Plato::scalarBoundaryDataAtIndex(tBoundaryData, tGlobalNodeOrdinal);
            const Plato::OrdinalType tIndex = std::abs(tCoordinates(3 * tGlobalNodeOrdinal + 1) + 5) * (kNumElemsPerEdge + 1) + std::abs(tCoordinates(3 * tGlobalNodeOrdinal + 2) + 5);
            tSidesetPressures(tIndex) = tPressure;
            tExpectedValue(tIndex) = tCoordinates(3 * tGlobalNodeOrdinal + 1) + tCoordinates(3 * tGlobalNodeOrdinal + 2) + 10;
        }
    });

    auto tHostSidesetPressures = Kokkos::create_mirror_view(tSidesetPressures);
    Kokkos::deep_copy(tHostSidesetPressures, tSidesetPressures);
    auto tHostExpectedValue = Kokkos::create_mirror_view(tExpectedValue);
    Kokkos::deep_copy(tHostExpectedValue, tExpectedValue);
    for(int i = 0; i < kNumNodesPerSide; ++i)
    {
        TEST_FLOATING_EQUALITY(tHostSidesetPressures(i), tHostExpectedValue(i), 1e-14);
    }
}
