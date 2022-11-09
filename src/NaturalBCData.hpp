#pragma once

#include "NaturalBCTypes.hpp"
#include "PlatoMathExpr.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "AbstractPlatoMeshIO.hpp"
#include "Plato_Utils.hpp"

#include <Teuchos_ParameterList.hpp>

#include <array>
#include <memory>

namespace Plato
{
using NaturalBCScalarData = Utils::NamedType<ScalarVector, struct NaturalBCScalarDataTag>;

/// Interface for data associated with a natural boundary condition.
/// The main purpose of this class is to provide methods for evaluating
/// a function on a boundary via the getScalarData and getVectorData members.
template<Plato::OrdinalType NumDofs>
class NaturalBCData
{
public:
    virtual ~NaturalBCData() = default;
    
    virtual std::unique_ptr<NaturalBCData> clone() const = 0;

    /// @return Vector boundary data at time @a aCurrentTime
    /// This is uniform over the entire boundary. 
    virtual Plato::Array<NumDofs> getVectorData(Plato::Scalar aCurrentTime = 0.0) const 
    {
        assert(false);
        return Plato::Array<NumDofs>{};
    }

    /// @return Scalar boundary data on the device associated with the data stored in @a aMeshIO.
    /// This is either uniform or spatially varying over a mesh, as given by the derived implementation.
    virtual NaturalBCScalarData getScalarData(const Plato::MeshIO& /*aMeshIO*/, const Plato::Scalar /*aCurrentTime*/) const
    {
        assert(false);
        return NaturalBCScalarData{ScalarVector{}};
    }

    NaturalBCScalarData getScalarData(const Plato::Scalar aCurrentTime) const
    {
        return getScalarData(nullptr, aCurrentTime);
    }

    NaturalBCScalarData getScalarData(const Plato::MeshIO& aMeshIO) const
    {
        constexpr double kDefaultTime = 0.0;
        return getScalarData(aMeshIO, kDefaultTime);
    }
};

/// Uniform in time and space boundary condition data.
template<Plato::OrdinalType NumDofs>
class UniformVectorNaturalBCData : public NaturalBCData<NumDofs>
{
public:
    explicit UniformVectorNaturalBCData(const Plato::Array<NumDofs>& aFlux) 
    : mFlux(aFlux)
    {}

    /// @throw std::runtime_error
    explicit UniformVectorNaturalBCData(const Teuchos::ParameterList& aSublist)
    {
        assert(aSublist.isType<Teuchos::Array<Plato::Scalar>>("Vector"));
        const auto& tFlux = aSublist.get<Teuchos::Array<Plato::Scalar>>("Vector");
        for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
        {
            mFlux(tDof) = tFlux[tDof];
        }
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        return std::make_unique<UniformVectorNaturalBCData<NumDofs>>(mFlux);
    }

    Plato::Array<NumDofs> getVectorData(
        const Plato::Scalar aCurrentTime) const override
    {
        return mFlux;
    }

private:
    Plato::Array<NumDofs> mFlux; /*!< force vector values */
};

/// Uniform in time and space boundary condition data.
template<Plato::OrdinalType NumDofs>
class UniformScalarNaturalBCData : public NaturalBCData<NumDofs>
{
public:
    explicit UniformScalarNaturalBCData(const Plato::Scalar aValue) 
    : mValue(aValue)
    {}

    /// @throw std::runtime_error
    explicit UniformScalarNaturalBCData(const Teuchos::ParameterList& aSublist)
    {
        assert(aSublist.isType<Plato::Scalar>("Value"));
        mValue = aSublist.get<Plato::Scalar>("Value");
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        return std::make_unique<UniformScalarNaturalBCData<NumDofs>>(mValue);
    }

    NaturalBCScalarData getScalarData(const Plato::MeshIO& /*aMeshIO*/, Plato::Scalar /*aCurrentTime*/) const override
    {
        ScalarVector tOutData("natural bc data", 1);
        auto tHostData = Kokkos::create_mirror_view(tOutData);
        tHostData(0) = mValue;
        Kokkos::deep_copy(tOutData, tHostData);
        return NaturalBCScalarData{std::move(tOutData)};
    }

private:
    Plato::Scalar mValue;
};

/// Non-uniform in time and uniform in space boundary condition data.
template<Plato::OrdinalType NumDofs>
class TimeVaryingVectorNaturalBCData : public NaturalBCData<NumDofs>
{
public:
    explicit TimeVaryingVectorNaturalBCData(
        const std::array<std::unique_ptr<Plato::MathExpr>, NumDofs>& aFluxExpr)
    {
        for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
        {
            assert(aFluxExpr[tDof]);
            mFluxExpr[tDof] = std::make_unique<Plato::MathExpr>(*aFluxExpr[tDof]);
        }
    }

    explicit TimeVaryingVectorNaturalBCData(const Teuchos::ParameterList& aSublist)
    {
        assert(aSublist.isType<Teuchos::Array<std::string>>("Vector"));
        const auto& tExpr = aSublist.get<Teuchos::Array<std::string>>("Vector");
        for(Plato::OrdinalType tDof=0; tDof<NumDofs; tDof++)
        {
            mFluxExpr[tDof] = std::make_unique<Plato::MathExpr>(tExpr[tDof]);
        }
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        return std::make_unique<TimeVaryingVectorNaturalBCData<NumDofs>>(mFluxExpr);
    }

    Plato::Array<NumDofs> getVectorData(
        const Plato::Scalar aCurrentTime) const override
    {
        Plato::Array<NumDofs> tFluxAtCurrentTime;
        for(int iDim = 0; iDim < NumDofs; ++iDim)
        {
            tFluxAtCurrentTime(iDim) = mFluxExpr[iDim]->value(aCurrentTime);
        }
        return tFluxAtCurrentTime;
    }

private:
    std::array<std::unique_ptr<Plato::MathExpr>, NumDofs> mFluxExpr;
};

/// Non-uniform in time and uniform in space boundary condition data.
template<Plato::OrdinalType NumDofs>
class TimeVaryingScalarNaturalBCData : public NaturalBCData<NumDofs>
{
public:
    explicit TimeVaryingScalarNaturalBCData(
        const Plato::MathExpr& aValueExpr)
        : mValueExpr(std::make_unique<Plato::MathExpr>(aValueExpr))
    {
    }

    explicit TimeVaryingScalarNaturalBCData(const Teuchos::ParameterList& aSublist)
    {
        assert(aSublist.isType<std::string>("Value"));
        const auto& tExpr = aSublist.get<std::string>("Value");
        mValueExpr = std::make_unique<Plato::MathExpr>(tExpr);
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        assert(mValueExpr);
        return std::make_unique<TimeVaryingScalarNaturalBCData<NumDofs>>(*mValueExpr);
    }

    NaturalBCScalarData getScalarData(const Plato::MeshIO& /*aMeshIO*/, Plato::Scalar aCurrentTime) const override
    {
        ScalarVector tOutData("natural bc data", 1);
        auto tHostData = Kokkos::create_mirror_view(tOutData);
        tHostData(0) = mValueExpr->value(aCurrentTime);
        Kokkos::deep_copy(tOutData, tHostData);
        return NaturalBCScalarData{std::move(tOutData)};
    }

private:
    std::unique_ptr<Plato::MathExpr> mValueExpr;
};

/// Non-uniform in space and uniform in time boundary condition data.
template<Plato::OrdinalType NumDofs>
class SpatiallyVaryingNaturalBCData : public NaturalBCData<NumDofs>
{
public:
    explicit SpatiallyVaryingNaturalBCData(std::string aVariableName)
    : mVariableName(std::move(aVariableName))
    {}

    explicit SpatiallyVaryingNaturalBCData(const Teuchos::ParameterList& aSublist)
    {
        if(!aSublist.isType<std::string>("Variable"))
        {
            ANALYZE_THROWERR(R"(Expected "Variable" field of string in variable pressure natural boundary condition.)");
        }
        mVariableName = aSublist.get<std::string>("Variable");
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        return std::make_unique<SpatiallyVaryingNaturalBCData<NumDofs>>(mVariableName);
    }

    NaturalBCScalarData getScalarData(const Plato::MeshIO& aMeshIO, Plato::Scalar /*aCurrentTime*/) const override
    {
        assert(aMeshIO->NumTimeSteps() > 0);
        return NaturalBCScalarData{aMeshIO->ReadNodeData(mVariableName, 0)};
    }

private:
    std::string mVariableName; // nodal field containing pressure values
};

template<Plato::OrdinalType NumDofs>
std::unique_ptr<NaturalBCData<NumDofs>> makeNaturalBCData(const Teuchos::ParameterList& aSublist)
{
    assert(aSublist.isParameter("Type"));
    switch(naturalBoundaryCondition(aSublist.get<std::string>("Type")))
    {
        case Neumann::UNIFORM_LOAD:
            if(aSublist.isType<Teuchos::Array<Plato::Scalar>>("Vector"))
            {
                return std::make_unique<UniformVectorNaturalBCData<NumDofs>>(aSublist);
            }
            else if(aSublist.isType<Teuchos::Array<std::string>>("Vector"))
            {
                return std::make_unique<TimeVaryingVectorNaturalBCData<NumDofs>>(aSublist);
            }
            else
            {
                ANALYZE_THROWERR(R"(Expected "Vector" field of type array of double or string in uniform natural boundary condition.)");
            }
            break;
        case Neumann::UNIFORM_PRESSURE:
            if(aSublist.isType<Plato::Scalar>("Value"))
            {
                return std::make_unique<UniformScalarNaturalBCData<NumDofs>>(aSublist);
            }
            else if(aSublist.isType<std::string>("Value"))
            {
                return std::make_unique<TimeVaryingScalarNaturalBCData<NumDofs>>(aSublist);
            }
            else 
            {
                ANALYZE_THROWERR(R"(Expected "Value" field of type double or string in uniform pressure natural boundary condition.)");
            }
            break;
        case Neumann::VARIABLE_PRESSURE:
            return std::make_unique<SpatiallyVaryingNaturalBCData<NumDofs>>(aSublist);
            break;
        default:
            ANALYZE_THROWERR("Unknown type encountered while constructing NaturalBCData.");
            break;
    }
}

/// The purpose of this function is to retrieve scalar boundary data in @a aBoundaryData 
/// associated with index @a aIndex for both spatially varying and uniform boundary
/// data types. 
KOKKOS_INLINE_FUNCTION
Scalar scalarBoundaryDataAtIndex(const NaturalBCScalarData& aBoundaryData, const OrdinalType aIndex)
{
    // Mod with size to support uniform data, which will be size 1
    return aBoundaryData.mValue(aIndex % aBoundaryData.mValue.size());
}
}