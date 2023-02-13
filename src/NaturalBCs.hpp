#pragma once

#include <vector>
#include <memory>
#include <type_traits>

#include "NaturalBC.hpp"

namespace Plato 
{

/***************************************************************************//**
 * \brief Owner class that contains a vector of NaturalBC objects.
 *
 * \tparam ElementType  Element type
 * \tparam NumDofs      number degrees of freedom per natural boundary condition force vector
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<
  typename ElementType,
  Plato::OrdinalType NumDofs=ElementType::mNumSpatialDims,
  Plato::OrdinalType DofsPerNode=NumDofs,
  Plato::OrdinalType DofOffset=0>
class NaturalBCs
{
// private member data
private:
    /*!< list of natural boundary condition */
    std::vector<std::shared_ptr<Plato::NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>> mBCs;

// private functions
private:
    /***************************************************************************//**
     * \brief Append natural boundary condition to natural boundary condition list.
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
    *******************************************************************************/
    void appendNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

    /***************************************************************************//**
     * \brief Return natural boundary condition type: uniform.
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
     *
     * \return shared pointer to an uniform natural boundary condition
    *******************************************************************************/
    std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>
    setLoadNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

    /***************************************************************************//**
     * \brief Return natural boundary condition type: uniform or variable pressure.
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
     *
     * \return shared pointer to an uniform pressure natural boundary condition
    *******************************************************************************/
    std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>
    setPressureNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

    /***************************************************************************//**
     * \brief Return natural boundary condition type: state function
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
     *
     * \return shared pointer to an uniform natural boundary condition
    *******************************************************************************/
    std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>
    setStateFunctionNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

    /***************************************************************************//**
     * \brief Return natural boundary condition type: state function
     *
     * \param  [in] aName    user-defined name for natural boundary condition sublist
     * \param  [in] aSubList natural boundary condition parameter sublist
     *
     * \return shared pointer to an uniform natural boundary condition
    *******************************************************************************/
    std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>
    setStefanBoltzmannNaturalBC(const std::string & aName, Teuchos::ParameterList &aSubList);

// public functions
public :
    /***************************************************************************//**
     * \brief Constructor that parses and creates a vector of NaturalBC objects.
     * \param [in] aParams input parameter list
    *******************************************************************************/
    NaturalBCs(Teuchos::ParameterList &aParams);

    /***************************************************************************//**
     * \brief Get the contribution to the assembled forcing vector from the owned
     * boundary conditions.
     *
     * \tparam StateScalarType   state forward automatically differentiated (FAD) type
     * \tparam ControlScalarType control FAD type
     * \tparam ConfigScalarType  configuration FAD type
     * \tparam ResultScalarType  result FAD type
     *
     * \param [in]  aState        2-D view of state variables.
     * \param [in]  aControl      2-D view of control variables.
     * \param [in]  aConfig       3-D view of configuration variables.
     * \param [out] aResult       Assembled vector to which the boundary terms will be added
     * \param [in]  aScale        scalar multiplier
     *
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get(
        const Plato::SpatialModel &,
        const Plato::ScalarMultiVectorT<  StateScalarType> &,
        const Plato::ScalarMultiVectorT<ControlScalarType> &,
        const Plato::ScalarArray3DT    < ConfigScalarType> &,
        const Plato::ScalarMultiVectorT< ResultScalarType> &,
              Plato::Scalar aScale = 1.0,
              Plato::Scalar aCurrentTime = 0.0) const;

    std::size_t numNaturalBCs() const;

    /// @pre @a aIndex must be less than numNaturalBCs
    const NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>& getNaturalBC(int aIndex) const;
};
// class NaturalBCs

/***************************************************************************//**
 * \brief NaturalBC::appendNaturalBC function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
void NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::appendNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    const auto tType = aSubList.get<std::string>("Type");
    const auto tNeumannType = Plato::naturalBoundaryCondition(tType);
    std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>> tBC;
    switch(tNeumannType)
    {
        case Plato::Neumann::UNIFORM_LOAD:
        case Plato::Neumann::VARIABLE_LOAD:
        {
            tBC = this->setLoadNaturalBC(aName, aSubList);
            break;
        }
        case Plato::Neumann::UNIFORM_PRESSURE:
        case Plato::Neumann::VARIABLE_PRESSURE:
        {
            tBC = this->setPressureNaturalBC(aName, aSubList);
            break;
        }
        case Plato::Neumann::STEFAN_BOLTZMANN:
        {
            tBC = this->setStefanBoltzmannNaturalBC(aName, aSubList);
            break;
        }
        default:
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition Type '" << tType.c_str() << "' is NOT supported.";
            ANALYZE_THROWERR(tMsg.str().c_str())
        }
    }
    mBCs.push_back(tBC);
}

namespace detail{
template<Plato::OrdinalType NumDofs, typename ValueType>
void setValuesFromIndexImpl(
    Teuchos::ParameterList &aSubList, const std::string& aParameterName, const ValueType& aZeroValue)
{
    const auto tDof = aSubList.get<Plato::OrdinalType>("Index", 0);
    Teuchos::Array<ValueType> tVector(NumDofs, aZeroValue);
    auto tValue = aSubList.get<ValueType>(aParameterName);
    tVector[tDof] = tValue;
    aSubList.set(aParameterName + "s", tVector);
    aSubList.remove(aParameterName);
    aSubList.remove("Index");
}
/// Converts a ParameterList containing a Value/Index pair to a full Values entry
/// @todo C++17 use constexpr if instead of enable_if
template<Plato::OrdinalType NumDofs, typename ValueType, 
    std::enable_if_t<std::is_same<ValueType, Plato::Scalar>::value, int> = 0>
void setValuesFromIndex(Teuchos::ParameterList &aSubList, const std::string& aParameterName)
{
    setValuesFromIndexImpl<NumDofs, ValueType>(aSubList, aParameterName, 0.0);
}
/// Converts a ParameterList containing a Value/Index pair to a full Values entry
/// @todo C++17 use constexpr if instead of enable_if
template<Plato::OrdinalType NumDofs, typename ValueType, 
    std::enable_if_t<std::is_same<ValueType, std::string>::value, int> = 0>
void setValuesFromIndex(Teuchos::ParameterList &aSubList, const std::string& aParameterName)
{
    setValuesFromIndexImpl<NumDofs, ValueType>(aSubList, aParameterName, "0.0");
}

/// @throw std::runtime_error Throws if @a aSubList contains multiple valid entries or no
///  valid entries. 
void affirmOneValidInput(const std::string & aName, Teuchos::ParameterList &aSubList);
}

/***************************************************************************//**
 * \brief NaturalBC::setLoadNaturalBC function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>
NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::setLoadNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    detail::affirmOneValidInput(aName, aSubList);

    const bool tBC_SingleValue = aSubList.isType<Plato::Scalar>("Value") 
                                || aSubList.isType<std::string>("Value")
                                || aSubList.isType<std::string>("Variable");

    if (tBC_SingleValue)
    {
        // For a single value, we expect Index to be specified
        // This converts that input into the full vector input
        if(aSubList.isType<Plato::Scalar>("Value"))
        {
            detail::setValuesFromIndex<NumDofs, Plato::Scalar>(aSubList, "Value");
        } 
        else if(aSubList.isType<std::string>("Value"))
        {
            detail::setValuesFromIndex<NumDofs, std::string>(aSubList, "Value");
        } 
        else if(aSubList.isType<std::string>("Variable"))
        {
            detail::setValuesFromIndex<NumDofs, std::string>(aSubList, "Variable");
        } 
    }

    return std::make_shared<Plato::NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>(aName, aSubList);
}

/***************************************************************************//**
 * \brief NaturalBC::setPressureNaturalBC function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>
NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::setPressureNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    detail::affirmOneValidInput(aName, aSubList);
    return std::make_shared<Plato::NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>(aName, aSubList);
}

/***************************************************************************//**
 * \brief NaturalBC::setStateFunctionNaturalBC function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>
NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::setStateFunctionNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    bool tBC_Value = aSubList.isType<std::string>("Value");

    bool tBC_Values = aSubList.isType<Teuchos::Array<std::string>>("Values");

    const auto tType = aSubList.get < std::string > ("Type");
    std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>> tBC;
    if (tBC_Values && tBC_Value)
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: Specify 'Values' OR 'Value' Parameter Keyword in "
            << "Parameter Sublist: '" << aName.c_str();
        ANALYZE_THROWERR(tMsg.str().c_str())
    }
    else if (tBC_Values)
    {
        auto tValues = aSubList.get<Teuchos::Array<std::string>>("Values");
        aSubList.set("Vector", tValues);
    }
    else if (tBC_Value)
    {

        auto tDof = aSubList.get<Plato::OrdinalType>("Index", 0);

        Teuchos::Array<std::string> tFluxVector(NumDofs, "0.0");
        auto tValue = aSubList.get<std::string>("Value");
        tFluxVector[tDof] = tValue;
        aSubList.set("Vector", tFluxVector);
    }
    else
    {
        std::stringstream tMsg;
        tMsg << "Natural Boundary Condition: Uniform Boundary Condition in Parameter Sublist: '"
            << aName.c_str() << "' was NOT parsed. Check input Parameter Keywords.";
        ANALYZE_THROWERR(tMsg.str().c_str())
    }

    tBC = std::make_shared<Plato::NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>(aName, aSubList);
    return tBC;
}

/***************************************************************************//**
 * \brief NaturalBC::setStefanBoltzmannNaturalBC function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::shared_ptr<NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>
NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::setStefanBoltzmannNaturalBC
(const std::string & aName, Teuchos::ParameterList &aSubList)
{
    return std::make_shared<Plato::NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>>(aName, aSubList);
}

/***************************************************************************//**
 * \brief NaturalBCs Constructor definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::NaturalBCs(Teuchos::ParameterList &aParams) :
mBCs()
{
    for (Teuchos::ParameterList::ConstIterator tItr = aParams.begin(); tItr != aParams.end(); ++tItr)
    {
        const Teuchos::ParameterEntry &tEntry = aParams.entry(tItr);
        if (!tEntry.isList())
        {
            ANALYZE_THROWERR("Natural Boundary Condition: Parameter in Boundary Conditions block not valid.  Expect lists only.")
        }

        const std::string &tName = aParams.name(tItr);
        if(aParams.isSublist(tName) == false)
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition: Sublist: '" << tName.c_str() << "' is NOT defined.";
            ANALYZE_THROWERR(tMsg.str().c_str())
        }
        Teuchos::ParameterList &tSubList = aParams.sublist(tName);

        if(tSubList.isParameter("Type") == false)
        {
            std::stringstream tMsg;
            tMsg << "Natural Boundary Condition: 'Type' Parameter Keyword in Parameter Sublist: '"
                << tName.c_str() << "' is NOT defined.";
            ANALYZE_THROWERR(tMsg.str().c_str())
        }

        this->appendNaturalBC(tName, tSubList);
    }
}

/***************************************************************************//**
 * \brief NaturalBCs::get function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::get(
     const Plato::SpatialModel                           & aSpatialModel,
     const Plato::ScalarMultiVectorT <  StateScalarType> & aState,
     const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
     const Plato::ScalarArray3DT     < ConfigScalarType> & aConfig,
     const Plato::ScalarMultiVectorT < ResultScalarType> & aResult,
           Plato::Scalar aScale,
           Plato::Scalar aCurrentTime
) const
{
    for (const auto &tMyBC : mBCs)
    {
        tMyBC->get(aSpatialModel, aState, aControl, aConfig, aResult, aScale, aCurrentTime);
    }
}

template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
std::size_t NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::numNaturalBCs() const
{
    return mBCs.size();
}

template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
const NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>& NaturalBCs<ElementType, NumDofs, DofsPerNode, DofOffset>::getNaturalBC(int aIndex) const
{
    return *mBCs.at(aIndex);
}
}
// namespace Plato

