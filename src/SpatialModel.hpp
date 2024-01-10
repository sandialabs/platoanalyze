#pragma once

#include <Teuchos_ParameterList.hpp>

#include "PlatoMesh.hpp"
#include "PlatoMask.hpp"
#include "ParseTools.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

#include "contact/ContactPair.hpp"
#include "contact/UpdateGraphForContact.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Base class for spatial domain
 */
class SpatialDomain
/******************************************************************************/
{
public:
    Plato::Mesh Mesh;     /*!< mesh database */

private:
    std::string         mElementBlockName;  /*!< element block name */
    std::string         mMaterialModelName; /*!< material model name */
    std::string         mSpatialDomainName; /*!< element block name */
    bool                mIsFixedBlock = false;      /*!< flag for fixed block */

    Plato::OrdinalVector mTotalElemLids;   /*!< List of local elements ids in this domain */
    Plato::OrdinalVector mMaskedElemLids;  /*!< List of local elements ids after application of a masked operation */

    Plato::DataMap mDataMap;

    bool mHasUniformBasis;

    // SpatialDomain isn't templated on SpatialDim, so allocate for 3D
    Plato::Matrix<3,3> mUniformCartesianBasis;

    bool mHasVaryingBasis;

    Plato::ScalarArray3D mVaryingCartesianBasis;

public:
    /******************************************************************************//**
     * \fn getDomainName
     * \brief Return domain name.
     * \return domain name
     **********************************************************************************/
    decltype(mSpatialDomainName) 
    getDomainName() const
    {
        return mSpatialDomainName;
    }

    /******************************************************************************//**
     * \fn setDomainName
     * \brief Set patial domain name.
     * \param [in] aName domain model name
     **********************************************************************************/
    void setDomainName(const std::string & aName)
    {
        mSpatialDomainName = aName;
    }

    /******************************************************************************//**
     * \fn getMaterialName
     * \brief Return material model name.
     * \return material model name
     **********************************************************************************/
    decltype(mMaterialModelName) 
    getMaterialName() const
    {
        return mMaterialModelName;
    }

    /******************************************************************************//**
     * \fn setMaterialName
     * \brief Set material model name.
     * \param [in] aName material model name
     **********************************************************************************/
    void setMaterialName(const std::string & aName)
    {
        mMaterialModelName = aName;
    }

    /******************************************************************************//**
     * \fn getElementBlockName
     * \brief Return element block name.
     * \return element block name
     **********************************************************************************/
    decltype(mElementBlockName) 
    getElementBlockName() const
    {
        return mElementBlockName;
    }

    /******************************************************************************//**
     * \fn setElementBlockName
     * \brief Set element block name.
     * \param [in] aName element block name
     **********************************************************************************/
    void setElementBlockName
    (const std::string & aName)
    {
        mElementBlockName = aName;
    }

    /******************************************************************************//**
     * \fn isFixedBlock
     * \brief Return whether block has fixed control field.
     **********************************************************************************/
    decltype(mIsFixedBlock) 
    isFixedBlock() const
    {
        return mIsFixedBlock;
    }

    /******************************************************************************//**
     * \fn numCells
     * \brief Return the number of cells.
     * \return number of cells
     **********************************************************************************/
    Plato::OrdinalType 
    numCells() const
    {
        return mMaskedElemLids.extent(0);
    }

    /******************************************************************************//**
     * \fn Plato::OrdinalType numNodes
     * \brief Returns the number of nodes in the mesh.
     * \return number of nodes
     **********************************************************************************/
    Plato::OrdinalType
    numNodes() const
    {
        return Mesh->NumNodes();
    }

    /******************************************************************************//**
     * \brief get cell ordinal list
     * Note: A const reference is returned to prevent the ref count from being modified.  
    **********************************************************************************/
    const Plato::OrdinalVector &
    cellOrdinals() const
    {
        return mMaskedElemLids;
    }

    /******************************************************************************//**
     * \fn cellOrdinals
     * \brief Set cell ordinals for this element block.
     * \param [in] aName element block name
     **********************************************************************************/
    void cellOrdinals(const std::string & aName)
    {
        this->setMaskLocalElemIDs(aName);
    }

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh        Default mesh
     * \param [in] aDataMap     Plato DataMap
     * \param [in] aInputParams Spatial model definition
     **********************************************************************************/
    SpatialDomain
    (      Plato::Mesh      aMesh,
           Plato::DataMap & aDataMap,
           std::string      aName);

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh        Default mesh
     * \param [in] aDataMap     Plato DataMap
     * \param [in] aInputParams Spatial model definition
     * \param [in] aName        Spatial model name
     **********************************************************************************/
    SpatialDomain
    (      Plato::Mesh              aMesh,
           Plato::DataMap         & aDataMap,
     const Teuchos::ParameterList & aInputParams,
           std::string              aName);

    /******************************************************************************//**
     * \brief Apply mask to this Domain
     *        This function removes elements that have a mask value of zero in \p aMask.
     *        Subsequent calls to numCells() and cellOrdinals() refer to the reduced list.
     *        Call removeMask() to remove the mask, or applyMask(...) to apply a different
     *        mask.
     * \param [in] aMask Plato Mask specifying active/inactive nodes and elements
    **********************************************************************************/
    template<Plato::OrdinalType mSpatialDim>
    void applyMask
    (std::shared_ptr<Plato::Mask<mSpatialDim>> aMask)
    {
        using OrdinalT = Plato::OrdinalType;

        auto tMask = aMask->cellMask();
        auto tTotalElemLids = mTotalElemLids;
        auto tNumEntries = tTotalElemLids.extent(0);

        // how many non-zeros in the mask?
        Plato::OrdinalType tSum(0);
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,tNumEntries), 
        KOKKOS_LAMBDA(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
        {
            auto tElemOrdinal = tTotalElemLids(aOrdinal);
            aUpdate += tMask(tElemOrdinal); 
        }, tSum);
        Kokkos::resize(mMaskedElemLids, tSum);

        auto tMaskedElemLids = mMaskedElemLids;

        // create a list of elements with non-zero mask values
        OrdinalT tOffset(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumEntries),
        KOKKOS_LAMBDA (const OrdinalT& aOrdinal, OrdinalT& aUpdate, const bool& tIsFinal)
        {
            auto tElemOrdinal = tTotalElemLids(aOrdinal);
            const OrdinalT tVal = tMask(tElemOrdinal);
            if( tIsFinal && tVal ) { tMaskedElemLids(aUpdate) = tElemOrdinal; }
            aUpdate += tVal;
        }, tOffset);
    }
        
    /******************************************************************************//**
     * \brief Remove applied mask.
     *        This function resets the element list in this domain to the original definition.
     *        If no mask has been applied, this function has no effect.
    **********************************************************************************/
    void
    removeMask();
    
    void setMaskLocalElemIDs
    (const std::string& aBlockName);

    void 
    initialize
    (const Teuchos::ParameterList & aInputParams);

    void
    parseUniformCartesianBasis(const Teuchos::ParameterList& aParamList);

    void
    parseVaryingCartesianBasis(const Teuchos::ParameterList& aParamList);

    // The cartesian basis is stored in the 3D matrix, mUniformCartesianBasis,
    // regardless of the actual dimension of the problem.  The accessors below
    // return only the relevant data for the requested dimension.
    inline void
    getUniformCartesianBasis(Plato::Matrix<3,3> & tBasis) const
    {
        tBasis = mUniformCartesianBasis;
    }

    inline void
    getUniformCartesianBasis(Plato::Matrix<2,2> & tBasis) const
    {
        for(int i=0; i<2; i++)
            for(int j=0; j<2; j++)
                tBasis(i,j) = mUniformCartesianBasis(i,j);
    }

    inline void
    getUniformCartesianBasis(Plato::Matrix<1,1> & tBasis) const
    {
        tBasis(0,0) = mUniformCartesianBasis(0,0);
    }

    inline void
    setUniformCartesianBasis(Plato::Matrix<3,3> const & tBasis)
    {
        mUniformCartesianBasis = tBasis;
    }

    inline void
    setUniformCartesianBasis(Plato::Matrix<2,2> const & tBasis)
    {
        for(int i=0; i<2; i++)
            for(int j=0; j<2; j++)
                mUniformCartesianBasis(i,j) = tBasis(i,j);
    }

    inline void
    setUniformCartesianBasis(Plato::Matrix<1,1> const & tBasis)
    {
        mUniformCartesianBasis(0,0) = tBasis(0,0);
    }

    bool hasUniformCartesianBasis() const
    { return mHasUniformBasis; }

    bool hasVaryingCartesianBasis() const
    { return mHasVaryingBasis; }

    inline
    Plato::ScalarArray3D
    getVaryingCartesianBasis() const
    {
        return mVaryingCartesianBasis;
    }

};
// class SpatialDomain

/******************************************************************************/
/*!
 \brief Spatial models contain the mesh, meshsets, domains, etc that define
 a discretized geometry.
 */
class SpatialModel
/******************************************************************************/
{
public:
    Plato::Mesh Mesh;     /*!< mesh database */
    std::vector<Plato::SpatialDomain> Domains; /*!< list of spatial domains, i.e. element blocks */

private:
    bool mHasContact;
    std::vector<Plato::Contact::ContactPair> mContactPairs;
    Plato::Contact::UpdateGraphForContact mUpdateGraphForContact;

public:
    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh     Default mesh
     **********************************************************************************/
    SpatialModel(Plato::Mesh aMesh);

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh Default mesh
     * \param [in] aInputParams Spatial model definition
     **********************************************************************************/
    SpatialModel(
              Plato::Mesh              aMesh,
        const Teuchos::ParameterList & aInputParams,
              Plato::DataMap         & aDataMap);

    template <Plato::OrdinalType mSpatialDim>
    void applyMask
    (std::shared_ptr<Plato::Mask<mSpatialDim>> aMask)
    {
        for( auto& tDomain : Domains )
        {
            tDomain.applyMask(aMask);
        }
    }

    /******************************************************************************//**
     * \brief Append spatial domain to spatial model.
     * \param [in] aDomain Spatial domain
     **********************************************************************************/
    void append
    (Plato::SpatialDomain & aDomain);

    void addContact(std::vector<Plato::Contact::ContactPair> aPairs);

    void NodeNodeGraph
    (Plato::OrdinalVector & aOffsetMap,
     Plato::OrdinalVector & aNodeOrds) const;

    void 
    NodeNodeGraphTranspose
    (Plato::OrdinalVector & aOffsetMap,
     Plato::OrdinalVector & aNodeOrds) const;

    std::vector<Plato::Contact::ContactPair>
    contactPairs() const { return mContactPairs; }

    bool
    hasContact() const { return mHasContact; }
};
// class SpatialModel

} // namespace Plato
