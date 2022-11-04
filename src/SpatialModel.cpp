#include "SpatialModel.hpp"

#include "PlatoMesh.hpp"
#include "PlatoMask.hpp"
#include "ParseTools.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoStaticsTypes.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{
SpatialDomain::SpatialDomain
(      Plato::Mesh      aMesh,
       Plato::DataMap & aDataMap,
 const std::string    & aName) :
    Mesh(aMesh),
    mDataMap(aDataMap),
    mSpatialDomainName(aName),
    mIsFixedBlock(false)
{}

SpatialDomain::SpatialDomain
(      Plato::Mesh              aMesh,
       Plato::DataMap         & aDataMap,
 const Teuchos::ParameterList & aInputParams,
 const std::string            & aName) :
    Mesh(aMesh),
    mDataMap(aDataMap),
    mSpatialDomainName(aName),
    mIsFixedBlock(false)
{
    this->initialize(aInputParams);
}

void
SpatialDomain::removeMask()
{
    Kokkos::deep_copy(mMaskedElemLids, mTotalElemLids);
}

void 
SpatialDomain::setMaskLocalElemIDs
(const std::string& aBlockName)
{
    auto tElemLids = Mesh->GetLocalElementIDs(aBlockName);
    auto tNumElems = tElemLids.size();
    mTotalElemLids = Plato::OrdinalVector("element list", tNumElems);
    mMaskedElemLids = Plato::OrdinalVector("masked element list", tNumElems);

    auto tTotalElemLids = mTotalElemLids;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumElems), KOKKOS_LAMBDA(const Plato::OrdinalType & aCellOrdinal)
    {
        tTotalElemLids(aCellOrdinal) = tElemLids[aCellOrdinal];
    }, "get element ids");
    Kokkos::deep_copy(mMaskedElemLids, mTotalElemLids);
}

void 
SpatialDomain::initialize
(const Teuchos::ParameterList & aInputParams)
{
    if(aInputParams.isType<std::string>("Element Block"))
    {
        mElementBlockName = aInputParams.get<std::string>("Element Block");
        this->cellOrdinals(mElementBlockName);
    }
    else
    {
        ANALYZE_THROWERR("Parsing new Domain. Required keyword 'Element Block' not found");
    }

    if(aInputParams.isType<std::string>("Material Model"))
    {
        mMaterialModelName = aInputParams.get<std::string>("Material Model");
    }
    else
    {
        ANALYZE_THROWERR("Parsing new Domain. Required keyword 'Material Model' not found");
    }
    if(aInputParams.isType<bool>("Fixed Control"))
    {
        mIsFixedBlock = aInputParams.get<bool>("Fixed Control");
    }

    this->setMaskLocalElemIDs(mElementBlockName);

    parseUniformCartesianBasis(aInputParams);
    parseVaryingCartesianBasis(aInputParams);
}

void
SpatialDomain::parseUniformCartesianBasis(const Teuchos::ParameterList& aParamList)
{
    if (aParamList.isSublist("Basis"))
    {
        if( Mesh->NumDimensions() == 3 )
        {
          Plato::ParseTools::getBasis(aParamList, mUniformCartesianBasis);
        }
        else
        if( Mesh->NumDimensions() == 2 )
        {
          Plato::Matrix<2,2> tBasis;
          Plato::ParseTools::getBasis(aParamList, tBasis);
          setUniformCartesianBasis(tBasis);
        }
        else
        if( Mesh->NumDimensions() == 1 )
        {
          Plato::Matrix<1,1> tBasis;
          Plato::ParseTools::getBasis(aParamList, tBasis);
          setUniformCartesianBasis(tBasis);
        }
        mHasUniformBasis = true;
    }
    else
    {
        mHasUniformBasis = false;
    }
}

void
SpatialDomain::parseVaryingCartesianBasis(const Teuchos::ParameterList& aParamList)
{
    if (aParamList.isType<std::string>("Basis Field"))
    {
        auto tBasisFieldName = aParamList.get<std::string>("Basis Field");
        mVaryingCartesianBasis = mDataMap.scalarArray3Ds[tBasisFieldName];
        mHasVaryingBasis = true;
    }
    else
    {
        mHasVaryingBasis = false;
    }
  
}

// The cartesian basis is stored in the 3D matrix, mUniformCartesianBasis,
// regardless of the actual dimension of the problem.  The accessors below
// return only the relevant data for the requested dimension.
inline void
SpatialDomain::getUniformCartesianBasis(Plato::Matrix<3,3> & tBasis) const
{
  tBasis = mUniformCartesianBasis;
}

inline void
SpatialDomain::getUniformCartesianBasis(Plato::Matrix<2,2> & tBasis) const
{
  for(int i=0; i<2; i++)
    for(int j=0; j<2; j++)
      tBasis(i,j) = mUniformCartesianBasis(i,j);
}

inline void
SpatialDomain::getUniformCartesianBasis(Plato::Matrix<1,1> & tBasis) const
{
  tBasis(0,0) = mUniformCartesianBasis(0,0);
}

inline void
SpatialDomain::setUniformCartesianBasis(Plato::Matrix<3,3> const & tBasis)
{
  mUniformCartesianBasis = tBasis;
}

inline void
SpatialDomain::setUniformCartesianBasis(Plato::Matrix<2,2> const & tBasis)
{
  for(int i=0; i<2; i++)
    for(int j=0; j<2; j++)
      mUniformCartesianBasis(i,j) = tBasis(i,j);
}

inline void
SpatialDomain::setUniformCartesianBasis(Plato::Matrix<1,1> const & tBasis)
{
  mUniformCartesianBasis(0,0) = tBasis(0,0);
}

inline
Plato::ScalarArray3D
SpatialDomain::getVaryingCartesianBasis() const
{
  return mVaryingCartesianBasis;
}

SpatialModel::SpatialModel(Plato::Mesh aMesh) : 
    Mesh(aMesh), 
    mHasContact(false), 
    mUpdateGraphForContact(aMesh) 
    {}

SpatialModel::SpatialModel(
          Plato::Mesh              aMesh,
    const Teuchos::ParameterList & aInputParams,
          Plato::DataMap         & aDataMap
) :
    Mesh(aMesh),
    mHasContact(false),
    mUpdateGraphForContact(aMesh) 
{
    if (aInputParams.isSublist("Spatial Model"))
    {
        auto tModelParams = aInputParams.sublist("Spatial Model");
        if (!tModelParams.isSublist("Domains"))
        {
            ANALYZE_THROWERR("Parsing 'Spatial Model' parameter list. Required 'Domains' parameter sublist not found");
        }

        auto tDomainsParams = tModelParams.sublist("Domains");
        for (auto tIndex = tDomainsParams.begin(); tIndex != tDomainsParams.end(); ++tIndex)
        {
            const auto &tEntry = tDomainsParams.entry(tIndex);
            const auto &tMyName = tDomainsParams.name(tIndex);

            if (!tEntry.isList())
            {
                ANALYZE_THROWERR("Parameter in 'Domains' parameter sublist within 'Spatial Model' parameter list not valid.  Expect lists only.");
            }

            Teuchos::ParameterList &tDomainParams = tDomainsParams.sublist(tMyName);
            Domains.push_back( { aMesh, aDataMap, tDomainParams, tMyName });
        }
    }
    else
    {
        ANALYZE_THROWERR("Parsing 'Plato Problem'. Required 'Spatial Model' parameter list not found");
    }
}

void 
SpatialModel::append
(Plato::SpatialDomain & aDomain)
{
    Domains.push_back(aDomain);
}

void 
SpatialModel::addContact
(const Plato::OrdinalVector & aChildNodes,
 const Plato::OrdinalVector & aParentElements)
{
    mUpdateGraphForContact.createNodeNodeGraph(aChildNodes, aParentElements);
}

void 
SpatialModel::returnNodeNodeGraph
(Plato::OrdinalVector & aOffsetMap,
 Plato::OrdinalVector & aNodeOrds)
 {
    if (mHasContact)
        mUpdateGraphForContact.getNodeNodeGraph(aOffsetMap, aNodeOrds);
    else
        Mesh->NodeNodeGraph(aOffsetMap, aNodeOrds);
 }

} // namespace Plato
