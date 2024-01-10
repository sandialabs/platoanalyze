#pragma once

#include "NaturalBCTypes.hpp"
#include "PlatoMathExpr.hpp"
#include "PlatoMathTypes.hpp"
#include "PlatoMesh.hpp"
#include "PlatoStaticsTypes.hpp"
#include "AbstractPlatoMesh.hpp"
#include "Plato_NamedType.hpp"

#include <Teuchos_ParameterList.hpp>

#include <array>
#include <memory>

namespace Plato
{
using NaturalBCScalarData = Utils::NamedType<ScalarVector, struct NaturalBCScalarDataTag>;

template<OrdinalType NumDofs>
using VectorDataView = Kokkos::View<Scalar*[NumDofs], Plato::MemSpace>;

template<OrdinalType NumDofs>
using NaturalBCVectorData = Utils::NamedType<VectorDataView<NumDofs>, struct NaturalBCVectorDataTag>;

constexpr static auto kValueParameterName = "Value";
constexpr static auto kValuesParameterName = "Values";
constexpr static auto kVariableParameterName = "Variable";
constexpr static auto kVariablesParameterName = "Variables";

enum struct BCDataType
{
    kScalar,
    kVector
};

namespace detail
{
template<OrdinalType ExpectedSize, typename T>
void affirmParameterSize(
    const Teuchos::ParameterList& aSublist, 
    const std::string& aParameterSingular, 
    const std::string& aParameterPlural);

/// @throw std::runtime_error if @a aVariableName cannot be found in the mesh.
ScalarVector getNodalData(const Plato::MeshIO& aMeshIO, const std::string& aVariableName);
}

/// Interface for data associated with a natural boundary condition.
/// The main purpose of this class is to provide methods for evaluating
/// a function on a boundary via the getScalarData and getVectorData members.
template<OrdinalType NumDofs>
class NaturalBCData
{
    static_assert(NumDofs > 0, "NumDofs must be greater than 0.");
public:
    virtual ~NaturalBCData() = default;
    
    virtual std::unique_ptr<NaturalBCData> clone() const = 0;

    /// @return Boundary data at time @a aCurrentTime. The array size is given by the member type `kNumComponents`
    /// @pre getDataType must return `BCDataType::kVector`
    virtual NaturalBCVectorData<NumDofs> getVectorData(const Plato::Mesh& aMesh, Scalar aCurrentTime) const = 0;

    NaturalBCVectorData<NumDofs> getVectorData(const Scalar aCurrentTime = 0.0) const 
    {
        return getVectorData(nullptr, aCurrentTime);
    }

    NaturalBCVectorData<NumDofs> getVectorData(const Plato::Mesh& aMesh) const 
    {
        constexpr double kDefaultTime = 0.0;
        return getVectorData(aMesh, kDefaultTime);
    }

    /// @return Scalar boundary data on the device associated with the data stored in @a aMesh.
    /// This is either uniform or spatially varying over a mesh, as given by the derived implementation.
    virtual NaturalBCScalarData getScalarData(const Plato::Mesh& aMesh, const Scalar aCurrentTime) const = 0;

    NaturalBCScalarData getScalarData(const Scalar aCurrentTime = 0.0) const
    {
        return getScalarData(nullptr, aCurrentTime);
    }

    NaturalBCScalarData getScalarData(const Plato::Mesh& aMesh) const
    {
        constexpr double kDefaultTime = 0.0;
        return getScalarData(aMesh, kDefaultTime);
    }

    virtual BCDataType getDataType() const = 0;
};

/// Uniform in time and space boundary condition data.
template<OrdinalType NumDofs, BCDataType DataType>
class UniformNaturalBCData : public NaturalBCData<NumDofs>
{
public:
    constexpr static OrdinalType kNumComponents = DataType == BCDataType::kScalar ? 1 : NumDofs;

    explicit UniformNaturalBCData(const std::array<Scalar, kNumComponents> aValues) 
        : mValues(std::move(aValues))
    {}

    /// @throw std::runtime_error
    explicit UniformNaturalBCData(const Teuchos::ParameterList& aSublist)
    {
        assert(aSublist.isType<Scalar>(kValueParameterName)
            || aSublist.isType<Teuchos::Array<Scalar>>(kValuesParameterName));
        detail::affirmParameterSize<kNumComponents, Scalar>(aSublist, kValueParameterName, kValuesParameterName);
        if(aSublist.isType<Scalar>(kValueParameterName))
        {
            mValues[0] = aSublist.get<Scalar>(kValueParameterName);
        }
        else if(aSublist.isType<Teuchos::Array<Scalar>>(kValuesParameterName))
        {
            const auto& tValues = aSublist.get<Teuchos::Array<Scalar>>(kValuesParameterName);
            std::copy(tValues.begin(), tValues.end(), std::begin(mValues));
        }
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        return std::make_unique<UniformNaturalBCData<NumDofs, DataType>>(mValues);
    }

    NaturalBCVectorData<NumDofs> getVectorData(const Plato::Mesh& /*aMesh*/, Scalar /*aCurrentTime*/) const override
    {
        assert(DataType == BCDataType::kVector);
        VectorDataView<NumDofs> tOutData("Natural BC data", 1);
        auto tHostData = Kokkos::create_mirror_view(tOutData);
        for(int i = 0; i < kNumComponents; ++i)
        {
            tHostData(0, i) = mValues[i];
        }
        Kokkos::deep_copy(tOutData, tHostData);
        return NaturalBCVectorData<NumDofs>{std::move(tOutData)};
    }

    NaturalBCScalarData getScalarData(const Plato::Mesh& /*aMesh*/, Scalar /*aCurrentTime*/) const override
    {
        ScalarVector tOutData("Natural BC data", 1);
        auto tHostData = Kokkos::create_mirror_view(tOutData);
        tHostData(0) = mValues[0];
        Kokkos::deep_copy(tOutData, tHostData);
        return NaturalBCScalarData{tOutData};
    }

    BCDataType getDataType() const override
    {
        return DataType;
    }

private:
    std::array<Scalar, kNumComponents> mValues;
};

/// Non-uniform in time and uniform in space boundary condition data.
template<OrdinalType NumDofs, BCDataType DataType>
class TimeVaryingNaturalBCData : public NaturalBCData<NumDofs>
{
public:
    constexpr static OrdinalType kNumComponents = DataType == BCDataType::kScalar ? 1 : NumDofs;

    explicit TimeVaryingNaturalBCData(
        const std::array<std::unique_ptr<Plato::MathExpr>, kNumComponents>& aExprs)
    {
        for(OrdinalType iDof = 0; iDof < kNumComponents; ++iDof)
        {
            assert(aExprs[iDof]);
            mExprs[iDof] = std::make_unique<Plato::MathExpr>(*aExprs[iDof]);
        }
    }

    explicit TimeVaryingNaturalBCData(const Teuchos::ParameterList& aSublist)
    {
        assert(aSublist.isType<std::string>(kValueParameterName) 
            || aSublist.isType<Teuchos::Array<std::string>>(kValuesParameterName));
        detail::affirmParameterSize<kNumComponents, std::string>(aSublist, kValueParameterName, kValuesParameterName);
        if(aSublist.isType<std::string>(kValueParameterName))
        {
            const auto& tExpr = aSublist.get<std::string>(kValueParameterName);
            mExprs[0] = std::make_unique<Plato::MathExpr>(tExpr);
        }
        else if(aSublist.isType<Teuchos::Array<std::string>>(kValuesParameterName))
        {
            const auto& tExprs = aSublist.get<Teuchos::Array<std::string>>(kValuesParameterName);
            for(OrdinalType iDof = 0; iDof < kNumComponents; ++iDof)
            {
                mExprs[iDof] = std::make_unique<Plato::MathExpr>(tExprs[iDof]);
            }
        }
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        return std::make_unique<TimeVaryingNaturalBCData<NumDofs, DataType>>(mExprs);
    }

    NaturalBCVectorData<NumDofs> getVectorData(const Plato::Mesh& /*aMesh*/, Scalar aCurrentTime) const override
    {
        assert(DataType == BCDataType::kVector);
        VectorDataView<NumDofs> tOutData("Natural BC data", 1);
        auto tHostData = Kokkos::create_mirror_view(tOutData);
        for(int i = 0; i < kNumComponents; ++i)
        {
            tHostData(0, i) = mExprs[i]->value(aCurrentTime);
        }
        Kokkos::deep_copy(tOutData, tHostData);
        return NaturalBCVectorData<NumDofs>{std::move(tOutData)};
    }

    NaturalBCScalarData getScalarData(const Plato::Mesh& /*aMesh*/, Scalar aCurrentTime) const override
    {
        ScalarVector tOutData("Natural BC data", 1);
        auto tHostData = Kokkos::create_mirror_view(tOutData);
        tHostData(0) = mExprs[0]->value(aCurrentTime);
        Kokkos::deep_copy(tOutData, tHostData);
        return NaturalBCScalarData{std::move(tOutData)};
    }

    BCDataType getDataType() const override
    {
        return DataType;
    }

private:
    std::array<std::unique_ptr<Plato::MathExpr>, kNumComponents> mExprs;
};

template<OrdinalType NumDofs, BCDataType DataType>
class SpatiallyVaryingNaturalBCData : public NaturalBCData<NumDofs>
{
public:
    constexpr static OrdinalType kNumComponents = DataType == BCDataType::kScalar ? 1 : NumDofs;

    explicit SpatiallyVaryingNaturalBCData(std::array<std::string, kNumComponents> aVariableNames)
    : mVariableNames(std::move(aVariableNames))
    {}

    explicit SpatiallyVaryingNaturalBCData(const Teuchos::ParameterList& aSublist)
    {
        assert(aSublist.isType<std::string>(kVariableParameterName) 
            || aSublist.isType<Teuchos::Array<std::string>>(kVariablesParameterName));
        detail::affirmParameterSize<kNumComponents, std::string>(aSublist, kVariableParameterName, kVariablesParameterName);
        if(aSublist.isType<std::string>(kVariableParameterName))
        {
            mVariableNames[0] = aSublist.get<std::string>(kVariableParameterName);
        } 
        else if(aSublist.isType<Teuchos::Array<std::string>>(kVariablesParameterName))
        {
            const auto tVariables = aSublist.get<Teuchos::Array<std::string>>(kVariablesParameterName);
            std::copy(tVariables.begin(), tVariables.end(), std::begin(mVariableNames));
        }
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        return std::make_unique<SpatiallyVaryingNaturalBCData<NumDofs, DataType>>(mVariableNames);
    }

    NaturalBCScalarData getScalarData(const Plato::Mesh& aMesh, Scalar /*aCurrentTime*/) const override
    {
        assert(aMesh);

        Plato::MeshIO tReader = Plato::MeshIOFactory::create(aMesh->FileName(), aMesh, "Read");
        return NaturalBCScalarData{detail::getNodalData(tReader, mVariableNames[0])};
    }

    NaturalBCVectorData<NumDofs> getVectorData(const Plato::Mesh& aMesh, Scalar /*aCurrentTime*/) const override
    {
        assert(aMesh);
        assert(DataType == BCDataType::kVector);

        Plato::MeshIO tReader = Plato::MeshIOFactory::create(aMesh->FileName(), aMesh, "Read");
        VectorDataView<NumDofs> tOutData("Natural BC data", aMesh->NumNodes());
        for(OrdinalType i = 0; i < kNumComponents; ++i)
        {
            Kokkos::deep_copy(Kokkos::subview(tOutData, Kokkos::ALL(), i), detail::getNodalData(tReader, mVariableNames[i]));
        }
        return NaturalBCVectorData<NumDofs>{std::move(tOutData)};
    }

    BCDataType getDataType() const override
    {
        return DataType;
    }

private:
    /// @pre Assumes that @a aSublist contains a parameter named `Variables` of type `Array(std::string)` or 
    ///  a parameter named `Variable` of type `std::string`.
    std::array<std::string, kNumComponents> mVariableNames; // nodal field containing load values
};

/// no-op for Stefan Boltzmann BC
template<OrdinalType NumDofs, BCDataType DataType=BCDataType::kScalar>
class StefanBoltzmannBCData : public NaturalBCData<NumDofs>
{
public:
    explicit StefanBoltzmannBCData(Scalar aStefanBoltzmannConstant)
    {
      mStefanBoltzmannConstant = aStefanBoltzmannConstant;
    }

    /// @throw std::runtime_error
    explicit StefanBoltzmannBCData(const Teuchos::ParameterList& aSublist)
    {
      constexpr static auto kStefanBoltzmannConstant = "Stefan Boltzmann Constant";
      
      if(aSublist.isType<Scalar>(kStefanBoltzmannConstant))
      {
        mStefanBoltzmannConstant = aSublist.get<Scalar>(kStefanBoltzmannConstant);
      }
      else
      {
        WARNING("'Stefan Boltzmann Constant' not provided.  Assuming MKS units.");
        mStefanBoltzmannConstant = 5.670374419e-8;
      }
    }

    std::unique_ptr<NaturalBCData<NumDofs>> clone() const override
    {
        return std::make_unique<StefanBoltzmannBCData<NumDofs, DataType>>(mStefanBoltzmannConstant);
    }

    NaturalBCVectorData<NumDofs> getVectorData(const Plato::Mesh& /*aMesh*/, Scalar /*aCurrentTime*/) const override
    {
      assert(DataType == BCDataType::kVector);
      VectorDataView<NumDofs> tOutData("Natural BC data", 1);
      return NaturalBCVectorData<NumDofs>{std::move(tOutData)};
    }

    NaturalBCScalarData getScalarData(const Plato::Mesh& /*aMesh*/, Scalar /*aCurrentTime*/) const override
    {
      ScalarVector tOutData("Natural BC data", 1);
      return NaturalBCScalarData{tOutData};
    }

    BCDataType getDataType() const override
    {
        return DataType;
    }

private:
    Scalar mStefanBoltzmannConstant;
};

template<OrdinalType NumDofs>
std::unique_ptr<NaturalBCData<NumDofs>> makeNaturalBCData(const Teuchos::ParameterList& aSublist)
{
    assert(aSublist.isParameter("Type"));
    switch(naturalBoundaryCondition(aSublist.get<std::string>("Type")))
    {
        case Neumann::UNIFORM_LOAD:
            if(aSublist.isType<Teuchos::Array<Scalar>>(kValuesParameterName))
            {
                return std::make_unique<UniformNaturalBCData<NumDofs, BCDataType::kVector>>(aSublist);
            }
            else if(aSublist.isType<Teuchos::Array<std::string>>(kValuesParameterName))
            {
                return std::make_unique<TimeVaryingNaturalBCData<NumDofs, BCDataType::kVector>>(aSublist);
            }
            else
            {
                ANALYZE_THROWERR(R"(Expected "Values" field of type array of double or string in uniform natural boundary condition.)");
            }
            break;
        case Neumann::VARIABLE_LOAD:
            if(aSublist.isType<std::string>(kVariableParameterName) || aSublist.isType<Teuchos::Array<std::string>>(kVariablesParameterName))
            {
                return std::make_unique<SpatiallyVaryingNaturalBCData<NumDofs, BCDataType::kVector>>(aSublist);
            }
            else
            {
                ANALYZE_THROWERR(R"(Expected "Variable" or "Variables" field of type string in variable load natural boundary condition.)");
            }
            break;
        case Neumann::UNIFORM_PRESSURE:
            if(aSublist.isType<Scalar>(kValueParameterName))
            {
                return std::make_unique<UniformNaturalBCData<NumDofs, BCDataType::kScalar>>(aSublist);
            }
            else if(aSublist.isType<std::string>(kValueParameterName))
            {
                return std::make_unique<TimeVaryingNaturalBCData<NumDofs, BCDataType::kScalar>>(aSublist);
            }
            else 
            {
                ANALYZE_THROWERR(R"(Expected "Value" field of type double or string in uniform pressure natural boundary condition.)");
            }
            break;
        case Neumann::VARIABLE_PRESSURE:
            if(aSublist.isType<std::string>(kVariableParameterName))
            {
                return std::make_unique<SpatiallyVaryingNaturalBCData<NumDofs, BCDataType::kScalar>>(aSublist);
            }
            else
            {
                ANALYZE_THROWERR(R"(Expected "Variable" field of type string in variable pressure natural boundary condition.)");
            }
            break;
        case Neumann::STEFAN_BOLTZMANN:
            return std::make_unique<StefanBoltzmannBCData<NumDofs>>(aSublist);
            break;
        default:
            ANALYZE_THROWERR("Unknown type encountered while constructing NaturalBCData.");
            break;
    }
}

namespace detail
{
template<OrdinalType ExpectedSize, typename T>
void affirmParameterSize(
    const Teuchos::ParameterList& aSublist, 
    const std::string& aParameterSingular, 
    const std::string& aParameterPlural)
{
    using ArrayType = Teuchos::Array<T>;
    const std::string tBaseMessage = "Natural boundary condition degrees of freedom does not"
        " match the number of boundary values provided for parameter ";
    if(aSublist.isType<T>(aParameterSingular) && ExpectedSize != 1)
    {
        const std::string tErrorMessage = tBaseMessage + aParameterSingular
            + ". Required: " + std::to_string(ExpectedSize) + ", received: 1";
        ANALYZE_THROWERR(tErrorMessage);
    }
    else if(aSublist.isType<ArrayType>(aParameterPlural) && ExpectedSize != aSublist.get<ArrayType>(aParameterPlural).size())
    {
        const int tSize = aSublist.get<ArrayType>(aParameterPlural).size();
        const std::string tErrorMessage = tBaseMessage + aParameterPlural
            + ". Required: " + std::to_string(ExpectedSize) + ", received: " + std::to_string(tSize);
        ANALYZE_THROWERR(tErrorMessage);
    }
}

}

/// The purpose of this function is to retrieve vector boundary data in @a aBoundaryData 
/// associated with index @a aIndex for both spatially varying and uniform boundary
/// data types. 
template<OrdinalType NumDofs>
KOKKOS_INLINE_FUNCTION
Array<NumDofs> vectorBoundaryDataAtIndex(const NaturalBCVectorData<NumDofs>& aBoundaryData, const OrdinalType aIndex)
{
    Array<NumDofs> tOutData;
    for(int i = 0; i < NumDofs; ++i)
    {
        // Mod with size to support uniform data, which will be size 1
        tOutData(i) = aBoundaryData.mValue(aIndex % aBoundaryData.mValue.extent(0), i);
    }
    return tOutData;
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
