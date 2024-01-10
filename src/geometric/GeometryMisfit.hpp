#pragma once

#include "geometric/WorksetBase.hpp"
#include "SurfaceArea.hpp"

#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "geometric/EvaluationTypes.hpp"
#include "geometric/AbstractScalarFunction.hpp"

#include "ExpInstMacros.hpp"

#include <fstream>
#include <ArborX.hpp>

namespace Plato
{
namespace GMF
{
    struct Points
    {
      double *d_x;
      double *d_y;
      double *d_z;
      int N;
    };
} // end namespace GMF
} // end namespace Plato

namespace Plato
{
namespace Geometric
{
enum Dim { X=0, Y, Z };

} // end namespace Geometric
} // end namespace Plato


namespace ArborX
{
template <>
struct AccessTraits<Plato::GMF::Points, PrimitivesTag>
{
  inline static std::size_t size(Plato::GMF::Points const &points) { return points.N; }
  KOKKOS_INLINE_FUNCTION static Point get(Plato::GMF::Points const &points, std::size_t i)
  {
    return {{(float)points.d_x[i], (float)points.d_y[i], (float)points.d_z[i]}};
  }
  using memory_space = Plato::MemSpace;
};
template <>
struct AccessTraits<Plato::GMF::Points, PredicatesTag>
{
  inline static std::size_t size(Plato::GMF::Points const &d) { return d.N; }
  KOKKOS_INLINE_FUNCTION static auto get(Plato::GMF::Points const &d, std::size_t i)
  {
    return nearest(Point{(float)d.d_x[i], (float)d.d_y[i], (float)d.d_z[i]}, 1);
  }
  using memory_space = Plato::MemSpace;
};
} // end namespace ArborX

namespace Plato
{

namespace Geometric
{


/******************************************************************************/
template<typename EvaluationType>
class GeometryMisfit :
    public EvaluationType::ElementType,
    public Plato::Geometric::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    using ElementType = typename EvaluationType::ElementType;

    using ElementType::mNumNodesPerCell;
    using ElementType::mNumNodesPerFace;
    using ElementType::mNumSpatialDims;

    using FunctionBaseType = typename Plato::Geometric::AbstractScalarFunction<EvaluationType>;

    using FunctionBaseType::mSpatialDomain;
    using FunctionBaseType::mDataMap;

    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::string mPointCloudName;
    std::string mPointCloudRowMapName;
    std::string mPointCloudColMapName;
    std::string mSideSetName;

  public:
    /**************************************************************************/
    GeometryMisfit(
        const Plato::SpatialDomain   & aSpatialDomain, 
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aFunctionParams, 
        const std::string            & aFunctionName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aFunctionParams, aFunctionName),
        mPointCloudName("")
    /**************************************************************************/
    {
        this->mHasBoundaryTerm = true;
        auto aCriterionParams = aFunctionParams.sublist("Criteria").sublist("Geometry Misfit");
        mSideSetName = aCriterionParams.get<std::string>("Sides");

        parsePointCloud(aCriterionParams);

        createPointGraph(aSpatialDomain.Mesh);
    }

    /**************************************************************************/
    void
    evaluate_conditional(
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult
    ) const override
    /**************************************************************************/
    { /* No-op.  Misfit is purely a boundary quantity */ }

    /**************************************************************************/
    void
    evaluate_boundary_conditional(
        const Plato::SpatialModel                          & aModel,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult
    ) const override
    /**************************************************************************/
    {

        // load the sideset specified in the input
        auto tElementOrds = aModel.Mesh->GetSideSetElements(mSideSetName);
        auto tNodeOrds    = aModel.Mesh->GetSideSetLocalNodes(mSideSetName);

        auto tOffsets = mDataMap.ordinalVectors[mPointCloudRowMapName];
        auto tIndices = mDataMap.ordinalVectors[mPointCloudColMapName];

        auto tPoints = mDataMap.scalarMultiVectors[mPointCloudName];

        // create functors
        Plato::SurfaceArea<ElementType> surfaceArea;

        Plato::OrdinalType tNumFaces = tElementOrds.size();
        Plato::ScalarVector tResultDenominator("denominator", tNumFaces);

        // for each point in the cloud, compute the square of the average normal distance to the nearest face
        auto tNumPoints = tOffsets.extent(0) - 1;
        Kokkos::parallel_for("compute misfit", Kokkos::RangePolicy<>(0,tNumPoints), KOKKOS_LAMBDA(const Plato::OrdinalType & aPointOrdinal)
        {
            // get face index
            auto tLocalFaceOrdinal = tIndices(tOffsets(aPointOrdinal));
            auto tElementOrdinal = tElementOrds(tLocalFaceOrdinal);

            // compute the map from local node id to global
            Plato::Array<mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
            for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<mNumNodesPerFace; tNodeOrd++)
            {
                tLocalNodeOrds[tNodeOrd] = tNodeOrds(tLocalFaceOrdinal*mNumNodesPerFace+tNodeOrd);
            }

            // get face centroid
            ConfigScalarType C_x = aConfig(tElementOrdinal, tLocalNodeOrds[0], Dim::X);
            ConfigScalarType C_y = aConfig(tElementOrdinal, tLocalNodeOrds[0], Dim::Y);
            ConfigScalarType C_z = aConfig(tElementOrdinal, tLocalNodeOrds[0], Dim::Z);
            for (Plato::OrdinalType tFaceVertI=1; tFaceVertI<mNumNodesPerFace; tFaceVertI++)
            {
                C_x += aConfig(tElementOrdinal, tLocalNodeOrds[tFaceVertI], Dim::X);
                C_y += aConfig(tElementOrdinal, tLocalNodeOrds[tFaceVertI], Dim::Y);
                C_z += aConfig(tElementOrdinal, tLocalNodeOrds[tFaceVertI], Dim::Z);
            }
            constexpr Plato::OrdinalType cNumNodesPerFace = mNumNodesPerFace;
            C_x /= cNumNodesPerFace;
            C_y /= cNumNodesPerFace;
            C_z /= cNumNodesPerFace;

            // get vertex 0 coordinates
            ConfigScalarType a_x = aConfig(tElementOrdinal, tLocalNodeOrds[0], Dim::X);
            ConfigScalarType a_y = aConfig(tElementOrdinal, tLocalNodeOrds[0], Dim::Y);
            ConfigScalarType a_z = aConfig(tElementOrdinal, tLocalNodeOrds[0], Dim::Z);

            // get vertex 1 coordinates
            ConfigScalarType b_x = aConfig(tElementOrdinal, tLocalNodeOrds[1], Dim::X);
            ConfigScalarType b_y = aConfig(tElementOrdinal, tLocalNodeOrds[1], Dim::Y);
            ConfigScalarType b_z = aConfig(tElementOrdinal, tLocalNodeOrds[1], Dim::Z);

            // vector, A, from centroid to vertex 0
            ConfigScalarType A_x = a_x - C_x;
            ConfigScalarType A_y = a_y - C_y;
            ConfigScalarType A_z = a_z - C_z;

            // vector, B, from centroid to vertex 1
            ConfigScalarType B_x = b_x - C_x;
            ConfigScalarType B_y = b_y - C_y;
            ConfigScalarType B_z = b_z - C_z;

            // unit normal vector, n = A X B / |A X B|
            ConfigScalarType n_x = A_y*B_z - A_z*B_y;
            ConfigScalarType n_y = A_z*B_x - A_x*B_z;
            ConfigScalarType n_z = A_x*B_y - A_y*B_x;
            ConfigScalarType mag_n = n_x*n_x + n_y*n_y + n_z*n_z;
            mag_n = sqrt(mag_n);
            n_x /= mag_n;
            n_y /= mag_n;
            n_z /= mag_n;

            // vector, P, from centroid to point
            ConfigScalarType P_x = tPoints(Dim::X, aPointOrdinal) - C_x;
            ConfigScalarType P_y = tPoints(Dim::Y, aPointOrdinal) - C_y;
            ConfigScalarType P_z = tPoints(Dim::Z, aPointOrdinal) - C_z;

            // normal distance to point
            ResultScalarType tDistance = P_x*n_x + P_y*n_y + P_z*n_z;

            // add to result
            Kokkos::atomic_add(&aResult(tElementOrdinal), tDistance);
            Kokkos::atomic_add(&tResultDenominator(tLocalFaceOrdinal), 1.0);

        });


        auto tCubatureWeights = ElementType::Face::getCubWeights();
        auto tCubaturePoints  = ElementType::Face::getCubPoints();
        auto tNumCubPoints = tCubatureWeights.size();

        Kokkos::parallel_for("divide and apply weight", Kokkos::RangePolicy<>(0,tNumFaces),
        KOKKOS_LAMBDA(const Plato::OrdinalType & aFaceOrdinal)
        {
            auto tElementOrdinal = tElementOrds(aFaceOrdinal);

            // compute the map from local node id to global
            Plato::Array<ElementType::mNumNodesPerFace, Plato::OrdinalType> tLocalNodeOrds;
            for( Plato::OrdinalType tNodeOrd=0; tNodeOrd<ElementType::mNumNodesPerFace; tNodeOrd++)
            {
                tLocalNodeOrds(tNodeOrd) = tNodeOrds(aFaceOrdinal*ElementType::mNumNodesPerFace+tNodeOrd);
            }

            ResultScalarType tWeight(0.0);
            for(Plato::OrdinalType iGpOrdinal=0; iGpOrdinal<tNumCubPoints; iGpOrdinal++)
            {
                auto tCubatureWeight = tCubatureWeights(iGpOrdinal);
                auto tCubaturePoint = tCubaturePoints(iGpOrdinal);
                auto tBasisGrads  = ElementType::Face::basisGrads(tCubaturePoint);

                ResultScalarType tSurfaceArea(0.0);
                surfaceArea(tElementOrdinal, tLocalNodeOrds, tBasisGrads, aConfig, tSurfaceArea);
                tWeight += tSurfaceArea * tCubatureWeight;
            }

            if (tResultDenominator(aFaceOrdinal) != 0)
            {
                aResult(tElementOrdinal) /= tResultDenominator(aFaceOrdinal);
            }
            aResult(tElementOrdinal) = tWeight * aResult(tElementOrdinal) * aResult(tElementOrdinal);

        });
    }

  public:

    void
    faceCentroids(
        Plato::Mesh              aMesh,
        Plato::ScalarMultiVector aCentroids
    )
    {
        auto tCoords       = aMesh->Coordinates();
        auto tConnectivity = aMesh->Connectivity();
        auto tElementOrds  = aMesh->GetSideSetElements(mSideSetName);
        auto tNodeOrds     = aMesh->GetSideSetLocalNodes(mSideSetName);

        const auto cNodesPerElement = aMesh->NumNodesPerElement();

        auto tNumFaces = tElementOrds.size();
        Kokkos::parallel_for("compute centroids", Kokkos::RangePolicy<>(0,tNumFaces), KOKKOS_LAMBDA(const Plato::OrdinalType & aFaceOrdinal)
        {
            for (Plato::OrdinalType tNodeI=0; tNodeI<mNumNodesPerFace; tNodeI++)
            {
                auto tElemLocalVertexOrdinal = tNodeOrds(aFaceOrdinal*mNumNodesPerFace+tNodeI);
                auto tMeshLocalVertexOrdinal = tConnectivity(tElementOrds(aFaceOrdinal)*cNodesPerElement + tElemLocalVertexOrdinal);
                for (Plato::OrdinalType tDimI=0; tDimI<mNumSpatialDims; tDimI++)
                {
                    aCentroids(tDimI, aFaceOrdinal) += tCoords(tMeshLocalVertexOrdinal * mNumSpatialDims + tDimI) / mNumNodesPerFace;
                }
            }

        });
    }

    void
    createPointGraph(
        Plato::Mesh aMesh
    )
    {
        std::stringstream tRowMapName;
        tRowMapName << mPointCloudName << "_rowmap";
        mPointCloudRowMapName = tRowMapName.str();

        std::stringstream tColMapName;
        tColMapName << mPointCloudName << "_colmap";
        mPointCloudColMapName = tColMapName.str();

        // only create the point cloud to face graph once
        if (mDataMap.ordinalVectors.count(mPointCloudRowMapName) == 0)
        {
            auto tElementOrds = aMesh->GetSideSetElements(mSideSetName);

            // create centroids
            Plato::ScalarMultiVector tCentroids("face centroids", mNumSpatialDims, tElementOrds.size());
            faceCentroids(aMesh, tCentroids);

            // construct search tree (this needs to be done in the constructor since the search result doesn't change)
            Plato::ScalarVector prim_x = Kokkos::subview(tCentroids, 0, Kokkos::ALL());
            Plato::ScalarVector prim_y = Kokkos::subview(tCentroids, 1, Kokkos::ALL());
            Plato::ScalarVector prim_z = Kokkos::subview(tCentroids, 2, Kokkos::ALL());
            Plato::OrdinalType tNumPrimitives = prim_x.extent(0);
            Plato::ExecSpace tExecSpace;
            ArborX::BVH<Plato::MemSpace>
            bvh{tExecSpace, Plato::GMF::Points{prim_x.data(), prim_y.data(), prim_z.data(), tNumPrimitives}};

            auto tPoints = mDataMap.scalarMultiVectors[mPointCloudName];
            Plato::ScalarVector pred_x = Kokkos::subview(tPoints, 0, Kokkos::ALL());
            Plato::ScalarVector pred_y = Kokkos::subview(tPoints, 1, Kokkos::ALL());
            Plato::ScalarVector pred_z = Kokkos::subview(tPoints, 2, Kokkos::ALL());
            Plato::OrdinalType tNumPredicates = pred_x.extent(0);

            Kokkos::View<int*, Plato::MemSpace> tIndices("indices", 0);
            Kokkos::View<int*, Plato::MemSpace> tOffsets("offsets", 0);

            ArborX::query(bvh, tExecSpace, Plato::GMF::Points{pred_x.data(), pred_y.data(), pred_z.data(), tNumPredicates}, tIndices, tOffsets);

            mDataMap.ordinalVectors[mPointCloudRowMapName] = tOffsets;
            mDataMap.ordinalVectors[mPointCloudColMapName] = tIndices;

            writeSearchResults(tPoints, tCentroids, tOffsets, tIndices);
        }
    }

    void
    writeSearchResults(
      const Plato::ScalarMultiVector                 & aPoints,
      const Plato::ScalarMultiVector                 & aCentroids,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aOffsets,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aIndices
    )
    {
        toVTK(/*fileBaseName=*/ "geometryMisfitPoints", aPoints);

        toVTK(/*fileBaseName=*/ "geometryMisfitCentroids", aCentroids);

        // compute vectors from aPoints to nearest aCentroid
        auto tNumPoints = aPoints.extent(1);
        auto tNumDims = aPoints.extent(0);
        Plato::ScalarMultiVector tVectors("vectors", tNumDims, tNumPoints);

        Kokkos::parallel_for("vectors", Kokkos::RangePolicy<>(0,tNumPoints), KOKKOS_LAMBDA(const Plato::OrdinalType & aPointOrdinal)
        {
            // TODO index directly into aIndices?
            auto tCentroidOrd = aIndices(aOffsets(aPointOrdinal));
            for(Plato::OrdinalType iDim=0; iDim<mNumSpatialDims; iDim++)
            {
                tVectors(iDim, aPointOrdinal) = aCentroids(iDim, tCentroidOrd) - aPoints(iDim, aPointOrdinal);
            }
        });

        // write vectors from points to nearest centroid
        toVTK(/*fileBaseName=*/ "geometryMisfitVectors", aPoints, tVectors);
    }

    void
    writeVTKHeader(
        std::ofstream & aOutFile
    )
    {
        aOutFile << "# vtk DataFile Version 2.0"  << std::endl;
        aOutFile << "Point data" << std::endl;
        aOutFile << "ASCII" << std::endl;
        aOutFile << "DATASET POLYDATA" << std::endl;
    }



    void
    toVTK(
            std::string                aBaseName,
      const Plato::ScalarMultiVector & aPoints
    )
    {
        std::ofstream tOutFile;
        tOutFile.open(aBaseName+".vtk");

        writeVTKHeader(tOutFile);

        auto tNumEntries = aPoints.extent(1);
        tOutFile << "POINTS " << tNumEntries << " float" << std::endl;

        writeToStream(tOutFile, aPoints);
    }

    void
    writeToStream(
            std::ostream             & aOutFile,
      const Plato::ScalarMultiVector & aEntries
    )
    {
        auto tEntries_Host = Kokkos::create_mirror_view(aEntries);
        Kokkos::deep_copy(tEntries_Host, aEntries);

        auto tNumDims = tEntries_Host.extent(0);
        auto tNumEntries = tEntries_Host.extent(1);
        for(Plato::OrdinalType iEntry=0; iEntry<tNumEntries; iEntry++)
        {
            for(Plato::OrdinalType iDim=0; iDim<tNumDims; iDim++)
            {
                aOutFile << tEntries_Host(iDim, iEntry) << " ";
            }
            aOutFile << std::endl;
        }
    }

    void
    toVTK(
            std::string                aBaseName,
      const Plato::ScalarMultiVector & aPoints,
      const Plato::ScalarMultiVector & aVectors
    )
    {
        std::ofstream tOutFile;
        tOutFile.open(aBaseName+".vtk");

        writeVTKHeader(tOutFile);

        auto tNumEntries = aPoints.extent(1);
        tOutFile << "POINTS " << tNumEntries << " float" << std::endl;
        writeToStream(tOutFile, aPoints);

        tOutFile << "POINT_DATA " << tNumEntries << std::endl;
        tOutFile << "VECTORS vectors float" << std::endl;
        writeToStream(tOutFile, aVectors);
    }

    void
    parsePointCloud(
        Teuchos::ParameterList & aFunctionParams
    )
    {
        mPointCloudName = aFunctionParams.get<std::string>("Point Cloud File Name");

        // only read the point cloud once
        if (mDataMap.scalarMultiVectors.count(mPointCloudName) == 0)
        {
            auto tHostPoints = this->readPointsFromFile(mPointCloudName);
            auto tDevicePoints = scalarMultiVectorFromData(tHostPoints);

            mDataMap.scalarMultiVectors[mPointCloudName] = tDevicePoints;
        }
    }

    inline std::vector<Plato::Scalar>
    readPoint(
        std::string tLineIn
    )
    {
        std::stringstream tStreamIn(tLineIn);
        std::vector<Plato::Scalar> tParsedValues;
        while (tStreamIn.good())
        {
            std::string tSubstr;
            getline(tStreamIn, tSubstr, ',');
            auto tNewValue = stod(tSubstr);
            tParsedValues.push_back(stod(tSubstr));
        }
        if (tParsedValues.size() != mNumSpatialDims)
        {
            ANALYZE_THROWERR("Error reading point cloud: line encountered with other than three values");
        }
        return tParsedValues;
    }

    std::vector<std::vector<Plato::Scalar>>
    readPointsFromFile(std::string aFileName)
    {
        std::vector<std::vector<Plato::Scalar>> tPoints;
        std::string tLineIn;
        std::ifstream tFileIn (mPointCloudName);
        if (tFileIn.is_open())
        {
            while (getline(tFileIn, tLineIn))
            {
                if (tLineIn.size() > 0 && tLineIn[0] != '#')
                {
                    auto tNewPoint = readPoint(tLineIn);
                    tPoints.push_back(tNewPoint);
                }
            }
            tFileIn.close();
        }
        else
        {
            ANALYZE_THROWERR("Failed to open point cloud file.");
        }
        return tPoints;
    }

    Plato::ScalarMultiVector
    scalarMultiVectorFromData(
        std::vector<std::vector<Plato::Scalar>> aPoints
    )
    {
        Plato::ScalarMultiVector tDevicePoints("point cloud", mNumSpatialDims, aPoints.size());
        auto tDevicePoints_Host = Kokkos::create_mirror_view(tDevicePoints);

        for (Plato::OrdinalType iPoint=0; iPoint<aPoints.size(); iPoint++)
        {
            for (Plato::OrdinalType iDim=0; iDim<mNumSpatialDims; iDim++)
            {
                tDevicePoints_Host(iDim, iPoint) = aPoints[iPoint][iDim];
            }
        }
        Kokkos::deep_copy(tDevicePoints, tDevicePoints_Host);

        return tDevicePoints;
    }
};

} // namespace Geometric

} // namespace Plato
