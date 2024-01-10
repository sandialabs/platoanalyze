#include <set>
#include <cstring>

#include "EngineMesh.hpp"
#include "alg/CrsMatrixUtils.hpp"

#include <KokkosSparse_SortCrs.hpp>

namespace Plato
{
    EngineMesh::EngineMesh(
        std::string aInputMeshName
    ) :
        mFileName(aInputMeshName),
        mCoordinates("node coordinates", 0),
        mConnectivity("element-node connectivity", 0),
        mNodeElementGraph_offsets("node-element graph offsets", 0),
        mNodeElementGraph_ordinals("node-element graph ordinals", 0),
        mNodeNodeGraph_offsets("node-node graph offsets", 0),
        mNodeNodeGraph_ordinals("node-node graph ordinals", 0)
    {
        initialize();
    }

    EngineMesh::~EngineMesh()
    {
        closeMesh();
    }

    void
    EngineMesh::initialize()
    {
        openMesh();

        auto tFaceGraph = mMeshIO->getFaceGraph(/*block id=*/ 0);  // EngineMesh requires all element blocks to have same element type
        auto tNumFacesPerElem = tFaceGraph.size();
        auto tNumNodesPerFace = tFaceGraph[0].size();
        Kokkos::resize(mFaceGraph, tNumFacesPerElem, tNumNodesPerFace);
        Kokkos::resize(mFaceGraphHost, tNumFacesPerElem, tNumNodesPerFace);
        auto tFaceGraphMirror = Kokkos::create_mirror_view(mFaceGraph);
        for( decltype(tNumFacesPerElem) iFace=0; iFace<tNumFacesPerElem; iFace++)
        {
            for( decltype(tNumNodesPerFace) iNode=0; iNode<tNumNodesPerFace; iNode++)
            {
                mFaceGraphHost(iFace, iNode) = tFaceGraph[iFace][iNode];
                tFaceGraphMirror(iFace, iNode) = tFaceGraph[iFace][iNode];
            }
        }
        Kokkos::deep_copy(mFaceGraph, tFaceGraphMirror);

        loadConnectivity();
        loadCoordinates();
        createNodeElementGraph();
        createNodeNodeGraph();
        createElementBlocks();
        loadSideSets();
        loadNodeSets();
    }

    void
    EngineMesh::closeMesh()
    {
        mMeshIO = nullptr;
    }

    void
    EngineMesh::openMesh()
    {
        mMeshIO = std::make_shared<ExodusIO>();
        mMeshIO->readMesh(mFileName, /*ignoreNodeMaps=*/false, /*ignoreElemMaps=*/false);

        auto tNumBlocks = mMeshIO->getNumElemBlks();
        mNumNodesPerElement = mMeshIO->getNnpeInBlk(/*blockIndex=*/0);
        for( decltype(tNumBlocks) tBlockIndex=1; tBlockIndex<tNumBlocks; tBlockIndex++ )
        {
            auto tNnpe = mMeshIO->getNnpeInBlk(tBlockIndex);
            if(tNnpe != mNumNodesPerElement)
            {
                throw std::runtime_error("Fatal Error: Encountered a mesh with multiple element types.");
            }
        }
        mElementType = mMeshIO->getElemTypeInBlk(/*blockIndex=*/0);
        mNumNodes = mMeshIO->getNumNodes();
        mNumElements = mMeshIO->getNumElems();
        mNumDimensions = mMeshIO->getDimensions();
    }

    void
    EngineMesh::loadConnectivity()
    {
        // determine length of connectivity array and resize mConnectivity
        auto tNumBlocks = mMeshIO->getNumElemBlks();
        Plato::OrdinalType tTotalNumConnect = 0;
        Plato::OrdinalType tTotalNumElements = 0;
        for( decltype(tNumBlocks) tBlockIndex=0; tBlockIndex<tNumBlocks; tBlockIndex++ )
        {
            auto tNnpe = mMeshIO->getNnpeInBlk(tBlockIndex);
            auto tNumElems = mMeshIO->getNumElemInBlk(tBlockIndex);
            tTotalNumConnect += tNnpe*tNumElems;
            tTotalNumElements += tNumElems;
        }
        Kokkos::resize(mConnectivity, tTotalNumConnect);

        // get connectivity from mMeshIO and write into mConnectivity;
        auto tHostConnectivity = Kokkos::create_mirror_view(mConnectivity);
        Plato::OrdinalType tStartingOrd = 0;
        for( decltype(tNumBlocks) tBlockIndex=0; tBlockIndex<tNumBlocks; tBlockIndex++ )
        {
            auto tNnpe = mMeshIO->getNnpeInBlk(tBlockIndex);
            auto tNumElems = mMeshIO->getNumElemInBlk(tBlockIndex);
            auto tNumConnect = tNnpe*tNumElems;
    
            auto tBlockConnect = mMeshIO->getElemToNodeConnInBlk(tBlockIndex);

            auto tTo = tHostConnectivity.data() + tStartingOrd;
            std::memcpy(tTo, tBlockConnect.data(), tNumConnect*sizeof(int));

            tStartingOrd += tNumConnect;
        }
        Kokkos::deep_copy(mConnectivity, tHostConnectivity);

        //createFullSurfaceSideSet(tHostConnectivity, tTotalNumElements);
    }

    void
    EngineMesh::loadCoordinates()
    {
        auto tMeshDim = mMeshIO->getDimensions();
        auto tNumNodes = mMeshIO->getNumNodes();
        Kokkos::resize(mCoordinates, tMeshDim*tNumNodes);
        auto tHostCoordinates = Kokkos::create_mirror_view(mCoordinates);

        auto tCoords = mMeshIO->getCoords();
        for( decltype(tNumNodes) tNodeIndex=0; tNodeIndex<tNumNodes; tNodeIndex++)
        {
            for( decltype(tMeshDim) tDimIndex=0; tDimIndex<tMeshDim; tDimIndex++)
            {
                tHostCoordinates(tNodeIndex*tMeshDim+tDimIndex) = tCoords[tDimIndex][tNodeIndex];
            }
        }
        Kokkos::deep_copy(mCoordinates, tHostCoordinates);
    }

    void
    EngineMesh::createNodeElementGraph()
    {
        auto tNumNodes = mMeshIO->getNumNodes();
        auto tNumElems = mMeshIO->getNumElems();

        auto tConnectivity = mConnectivity;
        auto tNumNodesPerElement = mNumNodesPerElement;

        Kokkos::resize(mNodeElementGraph_offsets, tNumNodes+1);

        Plato::OrdinalVector tNumConnectedElems("number of connected elements", tNumNodes+1);
        Kokkos::parallel_for("count connections", Kokkos::RangePolicy<>(0,tNumElems), KOKKOS_LAMBDA(Plato::OrdinalType aElemOrdinal)
        {
            for( decltype(tNumNodesPerElement) tElemLocalNodeOrd=0; tElemLocalNodeOrd<tNumNodesPerElement; tElemLocalNodeOrd++)
            {
                auto tProcLocalNodeOrd = tConnectivity(aElemOrdinal*tNumNodesPerElement+tElemLocalNodeOrd);
                Kokkos::atomic_increment(&tNumConnectedElems(tProcLocalNodeOrd));
            }
        });

        // compute offsets
        auto& tNodeElementGraph_offsets = mNodeElementGraph_offsets;

        Plato::OrdinalType tNumEntries(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumNodes+1),
        KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
        {
            const auto tVal = tNumConnectedElems(iOrdinal);
            if( tIsFinal )
            {
              tNodeElementGraph_offsets(iOrdinal) = aUpdate;
            }
            aUpdate += tVal;
        }, tNumEntries);

        Kokkos::resize(mNodeElementGraph_ordinals, tNumEntries);

        // set element ordinals
        auto& tNodeElementGraph_ordinals = mNodeElementGraph_ordinals;

        Kokkos::deep_copy(tNumConnectedElems, 0);
        Kokkos::parallel_for("element ordinals", Kokkos::RangePolicy<>(0, tNumElems), KOKKOS_LAMBDA(Plato::OrdinalType aElemOrdinal)
        {
            for( decltype(tNumNodesPerElement) tElemLocalNodeOrd=0; tElemLocalNodeOrd<tNumNodesPerElement; tElemLocalNodeOrd++)
            {
                auto tProcLocalNodeOrd = tConnectivity(aElemOrdinal*tNumNodesPerElement+tElemLocalNodeOrd);
                auto tStart = tNodeElementGraph_offsets(tProcLocalNodeOrd);
                auto tLocalOrd = Kokkos::atomic_fetch_add(&tNumConnectedElems(tProcLocalNodeOrd), 1);
                tNodeElementGraph_ordinals(tStart+tLocalOrd) = aElemOrdinal;
            }
        });

        // sort list of connected elements (otherwise cpu and gpu builds produce different graphs)
        KokkosSparse::sort_crs_graph<Plato::ExecSpace>(tNodeElementGraph_offsets, tNodeElementGraph_ordinals);
    }

    void
    EngineMesh::createNodeNodeGraph()
    {
        auto& tNodeElementGraph_offsets = mNodeElementGraph_offsets;
        auto& tNodeElementGraph_ordinals = mNodeElementGraph_ordinals;
        auto& tConnectivity = mConnectivity;
        auto tNumNodesPerElement = mNumNodesPerElement;

        auto tNumNodes = mMeshIO->getNumNodes();
        Kokkos::resize(mNodeNodeGraph_offsets, tNumNodes+1);
        auto tNumOrdinals = mNumNodesPerElement*tNodeElementGraph_ordinals.size();
        Plato::OrdinalVector tFatGraph_ordinals("node ordinals", tNumOrdinals);
        auto tNumEntries = tNumNodes+1;
        Plato::OrdinalVector tNumConnectedNodes("number of connected nodes", tNumEntries);
        Kokkos::parallel_for("node ordinals", Kokkos::RangePolicy<>(0, tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
        {
            Plato::OrdinalType tNumUnique(0);
            auto tFrom = tNodeElementGraph_offsets(aNodeOrdinal);
            auto tTo   = tNodeElementGraph_offsets(aNodeOrdinal+1);
            auto tFatGraphOffset = tFrom*tNumNodesPerElement;
            for( decltype(tFrom) tEntryIndex=tFrom; tEntryIndex<tTo; tEntryIndex++ )
            {
                auto tElemOrd = tNodeElementGraph_ordinals(tEntryIndex);
                auto tConnOffset = tElemOrd*tNumNodesPerElement;
                for( decltype(tElemOrd) tElemLocalNodeOrd=0; tElemLocalNodeOrd<tNumNodesPerElement; tElemLocalNodeOrd++ )
                {
                    auto tNodeOrd = tConnectivity(tConnOffset+tElemLocalNodeOrd);
                    bool isUnique = true;
                    for( decltype(tNumUnique) tIndex=0; tIndex<tNumUnique; tIndex++ )
                    {
                        if( tFatGraph_ordinals(tFatGraphOffset+tIndex) == tNodeOrd )
                        {
                            isUnique = false;
                        }
                    }
                    if(isUnique)
                    {
                        tFatGraph_ordinals(tFatGraphOffset+tNumUnique) = tNodeOrd;
                        tNumUnique++;
                    }
                }
            }
            tNumConnectedNodes(aNodeOrdinal) = tNumUnique;
        });

        // compute offsets
        auto& tNodeNodeGraph_offsets = mNodeNodeGraph_offsets;

        Plato::OrdinalType tNumNodeNodeEntries(0);

        Kokkos::parallel_scan (Kokkos::RangePolicy<>(0,tNumNodes+1),
        KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
        {
            const auto tVal = tNumConnectedNodes(iOrdinal);
            if( tIsFinal )
            {
              tNodeNodeGraph_offsets(iOrdinal) = aUpdate;
            }
            aUpdate += tVal;
        }, tNumNodeNodeEntries);

        Kokkos::resize(mNodeNodeGraph_ordinals, tNumNodeNodeEntries);

        auto& tNodeNodeGraph_ordinals = mNodeNodeGraph_ordinals;
        Kokkos::parallel_for("node ordinals", Kokkos::RangePolicy<>(0, tNumNodes), KOKKOS_LAMBDA(Plato::OrdinalType aNodeOrdinal)
        {
            auto tFrom = tNodeNodeGraph_offsets(aNodeOrdinal);
            auto tTo   = tNodeNodeGraph_offsets(aNodeOrdinal+1);

            auto tFatGraphFrom = tNodeElementGraph_offsets(aNodeOrdinal);
            auto tFatGraphOffset = tFatGraphFrom*tNumNodesPerElement;

            for( decltype(tFrom) tIndex=tFrom; tIndex<tTo; tIndex++ )
            {
                tNodeNodeGraph_ordinals(tIndex) = tFatGraph_ordinals(tFatGraphOffset++);
            }
        });

        // sort list of connected nodes (otherwise cpu and gpu builds produce different graphs)
        KokkosSparse::sort_crs_graph<Plato::ExecSpace>(tNodeNodeGraph_offsets, tNodeNodeGraph_ordinals);
    }

    void
    EngineMesh::createElementBlocks()
    {
        auto tNumBlocks = mMeshIO->getNumElemBlks();
        auto tNumElems = mMeshIO->getNumElems();

        auto& tBlockElementOrdinals = mBlockElementOrdinals;

        Plato::OrdinalType tElementOrdinalOffset(0);
        for( int tBlockIndex=0; tBlockIndex<tNumBlocks; tBlockIndex++ )
        {
            auto tBlockName = mMeshIO->getBlockName(tBlockIndex);
            auto tNumElemInBlk = mMeshIO->getNumElemInBlk(tBlockIndex);
            Plato::OrdinalVector tElementOrdinals("element ordinals", tNumElemInBlk);
            Kokkos::parallel_for("element ordinals", Kokkos::RangePolicy<>(0,tNumElemInBlk), KOKKOS_LAMBDA(Plato::OrdinalType aElemOrdinal)
            {
                tElementOrdinals(aElemOrdinal) = tElementOrdinalOffset+aElemOrdinal;
            });
            tBlockElementOrdinals[tBlockName] = tElementOrdinals;
            tElementOrdinalOffset += tNumElemInBlk;
        }
    }

    void
    EngineMesh::loadNodeSets()
    {
        auto tNumNodeSets = mMeshIO->getNumNodeSets();
        for( decltype(tNumNodeSets) tNodeSetIndex=0; tNodeSetIndex<tNumNodeSets; tNodeSetIndex++ )
        {
            auto tNodeOrdinals = mMeshIO->getNodeSetNodes(tNodeSetIndex);
            auto tNumNodesThisSet = tNodeOrdinals.size();

            Plato::OrdinalVector tDeviceData("node ordinals", tNumNodesThisSet);
            Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tNodeOrdinals.data(), tNumNodesThisSet);
            auto tDeviceView = Kokkos::create_mirror_view(tDeviceData);
            Kokkos::deep_copy(tDeviceView, tHostView);
            Kokkos::deep_copy(tDeviceData, tDeviceView);

            auto tName = mMeshIO->getNodeSetName(tNodeSetIndex);
            mNodeSetOrdinals[tName] = tDeviceData;
        }
    }

    void
    EngineMesh::loadSideSets()
    {
        auto tNumSideSets = mMeshIO->getNumSideSets();
        for( decltype(tNumSideSets) tSideSetIndex=0; tSideSetIndex<tNumSideSets; tSideSetIndex++ )
        {
            auto tNumNodesPerFace = mMeshIO->getSideSetNodesPerFace(tSideSetIndex);

            auto tElementOrdinals = mMeshIO->getSideSetElems(tSideSetIndex);
            auto tNumSidesThisSet = tElementOrdinals.size();

            Plato::OrdinalVector tElementOrds("element ordinals", tNumSidesThisSet);
            {
                Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tElementOrdinals.data(), tNumSidesThisSet);
                auto tDeviceView = Kokkos::create_mirror_view(tElementOrds);
                Kokkos::deep_copy(tDeviceView, tHostView);
                Kokkos::deep_copy(tElementOrds, tDeviceView);
            }

            auto tName = mMeshIO->getSideSetName(tSideSetIndex);
            mSideSetElementOrdinals[tName] = tElementOrds;

            auto tFaceOrdinals = mMeshIO->getSideSetFaces(tSideSetIndex);

            Plato::OrdinalVector tFaceOrds("face ordinals", tNumSidesThisSet);
            {
                Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tFaceOrdinals.data(), tNumSidesThisSet);
                auto tDeviceView = Kokkos::create_mirror_view(tFaceOrds);
                Kokkos::deep_copy(tDeviceView, tHostView);
                Kokkos::deep_copy(tFaceOrds, tDeviceView);
            }

            mSideSetFaceOrdinals[tName] = tFaceOrds;

            auto tFaceNodeOrdinals = mMeshIO->getSideSetNodes(tSideSetIndex);

            auto tNumFaceNodes =  tNumSidesThisSet*tNumNodesPerFace;

            // create node set from side set nodes
            std::set<Plato::OrdinalType> tUniqueNodes;
            for( decltype(tNumFaceNodes) tNodeI = 0; tNodeI < tNumFaceNodes; tNodeI++)
            {
                tUniqueNodes.insert(tFaceNodeOrdinals[tNodeI]);
            }
            auto tNumUniqueNodes = tUniqueNodes.size();
            std::vector<Plato::OrdinalType> tUniqueNodesVector;
            for( auto tUniqueNode : tUniqueNodes )
            {
                tUniqueNodesVector.push_back(tUniqueNode);
            }
            Plato::OrdinalVector tUniqueNodeOrds("unique node ordinals", tNumUniqueNodes);
            {
                Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tUniqueNodesVector.data(), tUniqueNodesVector.size());
                auto tDeviceView = Kokkos::create_mirror_view(tUniqueNodeOrds);
                Kokkos::deep_copy(tDeviceView, tHostView);
                Kokkos::deep_copy(tUniqueNodeOrds, tDeviceView);
            }
            mNodeSetOrdinals[tName] = tUniqueNodeOrds;


            Plato::OrdinalVector tNodeOrds("node ordinals", tNumFaceNodes);
            {
                Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tFaceNodeOrdinals.data(), tNumFaceNodes);
                auto tDeviceView = Kokkos::create_mirror_view(tNodeOrds);
                Kokkos::deep_copy(tDeviceView, tHostView);
                Kokkos::deep_copy(tNodeOrds, tDeviceView);
            }

            auto tNumNodesPerElement = mNumNodesPerElement;
            auto& tConnectivity = mConnectivity;
            Plato::OrdinalVector tLocalNodeOrds("local node ordinals", tNumFaceNodes);
            Kokkos::parallel_for("local node numbers", Kokkos::RangePolicy<>(0,tNumSidesThisSet), KOKKOS_LAMBDA(Plato::OrdinalType aSideOrdinal)
            {
                auto tElementOrd = tElementOrds(aSideOrdinal);
                for( decltype(tNumNodesPerFace) tNodeI = 0; tNodeI < tNumNodesPerFace; tNodeI++)
                {
                    for( Plato::OrdinalType tNodeJ = 0; tNodeJ < tNumNodesPerElement; tNodeJ++)
                    {
                        if( tNodeOrds(aSideOrdinal*tNumNodesPerFace+tNodeI) == tConnectivity(tElementOrd*tNumNodesPerElement + tNodeJ) )
                        {
                            tLocalNodeOrds(aSideOrdinal*tNumNodesPerFace+tNodeI) = tNodeJ;
                        }
                    }
                }
            });

            mSideSetLocalNodeOrdinals[tName] = tLocalNodeOrds;
        }
    }

    Plato::ScalarVectorT<const Plato::OrdinalType>
    EngineMesh::GetLocalElementIDs(
        std::string aBlockName
    ) const
    {
        return mBlockElementOrdinals.at(aBlockName);
    }

    std::vector<std::string>
    EngineMesh::GetElementBlockNames() const
    {
        std::vector<std::string> tRetVal;
        for(const auto& tPair : mBlockElementOrdinals)
        {
            tRetVal.push_back(tPair.first);
        }
        return tRetVal;
    }

    std::vector<std::string>
    EngineMesh::GetNodeSetNames() const
    {
        std::vector<std::string> tRetVal;
        for(const auto& tPair : mNodeSetOrdinals)
        {
            tRetVal.push_back(tPair.first);
        }
        return tRetVal;
    }

    std::vector<std::string>
    EngineMesh::GetSideSetNames() const
    {
        std::vector<std::string> tRetVal;
        for(const auto& tPair : mSideSetFaceOrdinals)
        {
            tRetVal.push_back(tPair.first);
        }
        return tRetVal;
    }

    std::string
    EngineMesh::FileName() const { return mFileName; }

    std::string
    EngineMesh::ElementType() const { return mElementType; }

    Plato::OrdinalType
    EngineMesh::NumNodes() const { return mNumNodes; }

    Plato::OrdinalType
    EngineMesh::NumNodesPerElement() const { return mNumNodesPerElement; }

    Plato::OrdinalType
    EngineMesh::NumElements() const { return mNumElements; }

    Plato::OrdinalType
    EngineMesh::NumDimensions() const { return mNumDimensions; }

    Plato::ScalarVectorT<const Plato::Scalar>
    EngineMesh::Coordinates() const
    {
        return mCoordinates;
    }

    void
    EngineMesh::SetCoordinates(
        Plato::ScalarVector aCoordinates
    )
    {
        if( mCoordinates.size() != aCoordinates.size() )
        {
            throw std::runtime_error("Dimension mismatch.  Failed to set coordinates");
        }
        Kokkos::deep_copy(mCoordinates, aCoordinates);
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    EngineMesh::Connectivity()
    {
        return mConnectivity;
    }

    void
    EngineMesh::NodeNodeGraph(
        Plato::OrdinalVectorT<Plato::OrdinalType> & aOffsetMap,
        Plato::OrdinalVectorT<Plato::OrdinalType> & aNodeOrds
    )
    {
        aOffsetMap = mNodeNodeGraph_offsets;
        aNodeOrds = mNodeNodeGraph_ordinals;
    }

    void
    EngineMesh::NodeElementGraph(
        Plato::OrdinalVectorT<const Plato::OrdinalType> & aOffsetMap,
        Plato::OrdinalVectorT<const Plato::OrdinalType> & aElementOrds
    )
    {
        aOffsetMap = mNodeElementGraph_offsets;
        aElementOrds = mNodeElementGraph_ordinals;
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    EngineMesh::GetSideSetFaces( std::string aSideSetName) const
    {
        return mSideSetFaceOrdinals.at(aSideSetName);
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    EngineMesh::GetSideSetElements( std::string aSideSetName) const
    {
        return mSideSetElementOrdinals.at(aSideSetName);
    }

    /* element-local indexing (not global) */
    Plato::OrdinalVectorT<const Plato::OrdinalType>
    EngineMesh::GetSideSetLocalNodes( std::string aSideSetName) const
    {
        return mSideSetLocalNodeOrdinals.at(aSideSetName);
    }
    
    Plato::OrdinalVectorT<const Plato::OrdinalType>
    EngineMesh::GetSideSetElementsComplement( std::vector<std::string> aExcludeNames)
    {
        createFullSurfaceSideSet();

        if( aExcludeNames.size() == 0 )
        {
            return mFullSurfaceSideSetElementOrdinals;
        }

        if( mSideSetElementsComplementOrdinals.count(aExcludeNames) )
        {
            return mSideSetElementsComplementOrdinals.at(aExcludeNames);
        }
        else
        {
            createComplement( aExcludeNames );
            return mSideSetElementsComplementOrdinals.at(aExcludeNames);
        }
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    EngineMesh::GetSideSetFacesComplement( std::vector<std::string> aExcludeNames)
    {
        createFullSurfaceSideSet();

        if( aExcludeNames.size() == 0 )
        {
            return mFullSurfaceSideSetFaceOrdinals;
        }

        if( mSideSetFacesComplementOrdinals.count(aExcludeNames) )
        {
            return mSideSetFacesComplementOrdinals.at(aExcludeNames);
        }
        else
        {
            createComplement( aExcludeNames );
            return mSideSetFacesComplementOrdinals.at(aExcludeNames);
        }
    }

    /* element-local indexing (not global) */
    Plato::OrdinalVectorT<const Plato::OrdinalType>
    EngineMesh::GetSideSetLocalNodesComplement( std::vector<std::string> aExcludeNames)
    {
        createFullSurfaceSideSet();

        if( aExcludeNames.size() == 0 )
        {
            return mFullSurfaceSideSetLocalNodeOrdinals;
        }

        if( mSideSetLocalNodesComplementOrdinals.count(aExcludeNames) )
        {
            return mSideSetLocalNodesComplementOrdinals.at(aExcludeNames);
        }
        else
        {
            createComplement( aExcludeNames );
            return mSideSetLocalNodesComplementOrdinals.at(aExcludeNames);
        }
    }
    
    void
    EngineMesh::createComplement( std::vector<std::string> aExcludeNames )
    {
        auto tNumFacesPerElement = mFaceGraph.extent(0);
        auto tNumNodesPerFace = mFaceGraph.extent(1);

        // expand FullSurface side set into indexable array
        auto tNumTotalFaces = mNumElements*tNumFacesPerElement;
        Plato::OrdinalVector tFullSurfaceArray("face mask", tNumTotalFaces);
        Plato::OrdinalVector tRemainingSurfaceArray("face mask", tNumTotalFaces);

        auto tFaces = mFullSurfaceSideSetFaceOrdinals;
        auto tElements = mFullSurfaceSideSetElementOrdinals;

        auto tNumFaces = tFaces.size();
        Kokkos::parallel_for("create indexable array", Kokkos::RangePolicy<>(0,tNumFaces), KOKKOS_LAMBDA(Plato::OrdinalType aFaceOrdinal)
        {
            tFullSurfaceArray(tElements(aFaceOrdinal)*tNumFacesPerElement + tFaces(aFaceOrdinal)) = 1;
        });

        Kokkos::deep_copy(tRemainingSurfaceArray, tFullSurfaceArray);

        // subtract aExcludeNames from FullSurface
        auto tNumExcludes = aExcludeNames.size();
        for( const auto& tExcludeName : aExcludeNames )
        {
            auto tFaces = mSideSetFaceOrdinals[tExcludeName];
            auto tElements = mSideSetElementOrdinals[tExcludeName];

            auto tNumFaces = tFaces.size();
            Kokkos::parallel_for("subtract", Kokkos::RangePolicy<>(0,tNumFaces), KOKKOS_LAMBDA(Plato::OrdinalType aFaceOrdinal)
            {
                tRemainingSurfaceArray(tElements(aFaceOrdinal)*tNumFacesPerElement + tFaces(aFaceOrdinal)) = 0;
            });
        }

        // count remaining sides and create resulting views
        Plato::OrdinalType tNumRemainingFaces(0);
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumTotalFaces),
        KOKKOS_LAMBDA(const Plato::OrdinalType& aFaceOrdinal, Plato::OrdinalType & aUpdate)
        {
            if ( tRemainingSurfaceArray(aFaceOrdinal) == 1 )
            {
                Kokkos::atomic_increment(&aUpdate);
            }
        }, tNumRemainingFaces);

        // parallel_scan on RemainingSurface and write into new side set elements and faces
        Plato::OrdinalVector tNewComplementFaces("complement faces", tNumRemainingFaces);
        Plato::OrdinalVector tNewComplementElements("complement elements", tNumRemainingFaces);
        Plato::OrdinalVector tNewComplementLocalNodes("complement local nodes", tNumRemainingFaces*tNumNodesPerFace);
        Plato::OrdinalType tOffset(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumTotalFaces),
        KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
        {
            const Plato::OrdinalType tVal = tRemainingSurfaceArray(iOrdinal);
            if( tIsFinal && tVal )
            {
                auto tFaceOrdinal = iOrdinal % tNumFacesPerElement;
                auto tElementOrdinal = iOrdinal / tNumFacesPerElement;
                tNewComplementFaces(aUpdate) = tFaceOrdinal;
                tNewComplementElements(aUpdate) = tElementOrdinal;
            }
            aUpdate += tVal;
        }, tOffset);

        // populate local nodes 
        auto tFaceGraph = mFaceGraph;
        Kokkos::parallel_for("local nodes", Kokkos::RangePolicy<>(0,tNumRemainingFaces), KOKKOS_LAMBDA(Plato::OrdinalType aFaceOrdinal)
        {
            auto tOffset = aFaceOrdinal*tNumNodesPerFace;
            auto tLocalFaceOrdinal = tNewComplementFaces(aFaceOrdinal);
            for( decltype(tNumNodesPerFace) tNodeOrdinal=0; tNodeOrdinal<tNumNodesPerFace; tNodeOrdinal++)
            {
                tNewComplementLocalNodes(tOffset+tNodeOrdinal) = tFaceGraph(tLocalFaceOrdinal, tNodeOrdinal);
            }
        });

        // add to Complements maps
        mSideSetFacesComplementOrdinals[aExcludeNames] = tNewComplementFaces;
        mSideSetElementsComplementOrdinals[aExcludeNames] = tNewComplementElements;
        mSideSetLocalNodesComplementOrdinals[aExcludeNames] = tNewComplementLocalNodes;
    }

    void
    EngineMesh::createFullSurfaceSideSet(
    )
    {
        if( mFullSurfaceSideSetElementOrdinals.size() )
        {
            // already computed.
            return;
        }

        // this function is called once if the helmholtz filter uses surface correction.  The
        // full surface side set is easy to compute on the host and reasonably fast, roughly
        // three seconds for a 1e6 element mesh.  If this becomes a bottleneck, this could be 
        // rewritten to run on the device.

        auto tConnectivityHost = Kokkos::create_mirror_view(mConnectivity);
        Kokkos::deep_copy(tConnectivityHost, mConnectivity);

        auto tNumFacesPerElement = mFaceGraph.extent(0);
        auto tNumNodesPerFace = mFaceGraph.extent(1);

        using FaceKey = std::vector<Plato::OrdinalType>;
        using ElemFacePairList = std::vector<std::pair<Plato::OrdinalType, Plato::OrdinalType>>;
        std::map<FaceKey, ElemFacePairList> tConnections;

        // for each element
        for( decltype(mNumElements) iElement = 0; iElement < mNumElements; iElement++ )
        {
            // for each element face
            for( decltype(tNumFacesPerElement) iFace = 0; iFace < tNumFacesPerElement; iFace++ )
            {
                auto tConnOffset = iElement*mNumNodesPerElement;
                FaceKey tFace(tNumNodesPerFace);

                // get face nodes
                for( decltype(tNumNodesPerFace) iNode = 0; iNode < tNumNodesPerFace; iNode++ )
                {
                    tFace[iNode] = tConnectivityHost(tConnOffset+mFaceGraphHost(iFace, iNode));
                }
             
                // sort face nodes
                std::sort(tFace.begin(), tFace.end());

                // add elem/face pair to map
                if( tConnections.count(tFace) > 0 )
                {
                    tConnections[tFace].push_back({iElement,iFace});
                }
                else
                {
                    tConnections[tFace] = ElemFacePairList();
                    tConnections[tFace].push_back({iElement,iFace});
                }
            }
        }

        // find map entries with only one attached element and add to new side set
        std::vector<Plato::OrdinalType> tSideSetElements, tSideSetFaces, tSideSetNodes;
        for( const auto& tMapEntry : tConnections )
        {
            const auto& tElemFacePairs = tMapEntry.second;
            if( tElemFacePairs.size() == 1 )
            {
                auto tElement = tElemFacePairs[0].first;
                auto tLocalFace = tElemFacePairs[0].second;
                tSideSetElements.push_back(tElement);
                tSideSetFaces.push_back(tLocalFace);
                for( decltype(tNumNodesPerFace) iNode = 0; iNode < tNumNodesPerFace; iNode++ )
                {
                    tSideSetNodes.push_back(mFaceGraphHost(tLocalFace, iNode));
                }
            }
        }

        Kokkos::resize(mFullSurfaceSideSetElementOrdinals, tSideSetElements.size());
        {
            Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tSideSetElements.data(), tSideSetElements.size());
            auto tDeviceView = Kokkos::create_mirror_view(mFullSurfaceSideSetElementOrdinals);
            Kokkos::deep_copy(tDeviceView, tHostView);
            Kokkos::deep_copy(mFullSurfaceSideSetElementOrdinals, tDeviceView);
        }

        Kokkos::resize(mFullSurfaceSideSetFaceOrdinals, tSideSetFaces.size());
        {
            Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tSideSetFaces.data(), tSideSetFaces.size());
            auto tDeviceView = Kokkos::create_mirror_view(mFullSurfaceSideSetFaceOrdinals);
            Kokkos::deep_copy(tDeviceView, tHostView);
            Kokkos::deep_copy(mFullSurfaceSideSetFaceOrdinals, tDeviceView);
        }

        Kokkos::resize(mFullSurfaceSideSetLocalNodeOrdinals, tSideSetNodes.size());
        {
            Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tSideSetNodes.data(), tSideSetNodes.size());
            auto tDeviceView = Kokkos::create_mirror_view(mFullSurfaceSideSetLocalNodeOrdinals);
            Kokkos::deep_copy(tDeviceView, tHostView);
            Kokkos::deep_copy(mFullSurfaceSideSetLocalNodeOrdinals, tDeviceView);
        }
    }

    Plato::OrdinalVectorT<const Plato::OrdinalType>
    EngineMesh::GetNodeSetNodes( std::string aNodeSetName) const
    {
        return mNodeSetOrdinals.at(aNodeSetName);
    }

    void
    EngineMesh::CreateNodeSet(
        std::string aNodeSetName,
        std::initializer_list<Plato::OrdinalType> aNodes
    )
    {
        if( mNodeSetOrdinals.count(aNodeSetName) )
        {
            ANALYZE_THROWERR("Node set already exists.  Names must be unique.");
        }
        std::vector<Plato::OrdinalType> tNodesHost(aNodes);
        auto tNumNodes = tNodesHost.size();
        Plato::OrdinalVector tDeviceData("node ordinals", tNumNodes);
        Kokkos::View<Plato::OrdinalType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tNodesHost.data(), tNumNodes);
        auto tDeviceView = Kokkos::create_mirror_view(tDeviceData);
        Kokkos::deep_copy(tDeviceView, tHostView);
        Kokkos::deep_copy(tDeviceData, tDeviceView);

        mNodeSetOrdinals[aNodeSetName] = tDeviceData;
    }
}
