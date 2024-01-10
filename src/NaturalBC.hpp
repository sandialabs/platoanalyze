/*
 * NaturalBC.hpp
 *
 *  Created on: Mar 15, 2020
 */

#pragma once

#include <sstream>

#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoMathExpr.hpp"
#include "NaturalBCTypes.hpp"
#include "NaturalBCData.hpp"
#include "SurfaceLoadIntegral.hpp"
#include "SurfacePressureIntegral.hpp"
#include "StefanBoltzmann.hpp"

namespace Plato
{
/// @return The string associated with parameter @a aParameterName read from sublist
///  @a aSublist. @a aBCName is the name of the parent list used for error messages.
/// @throw std::runtime_error if @a aParameterName cannot be found in @a aSublist or
///  it does not have type string.
std::string getStringDataAndAffirmExists(
    const std::string& aParameterName, const Teuchos::ParameterList& aSublist, const std::string& aBCName);

/// @throw std::runtime_error if @a aParameterName cannot be found in @a aSublist
void affirmExists(
    const std::string& aParameterName, const Teuchos::ParameterList& aSublist, const std::string& aBCName);

/***************************************************************************//**
 * \brief Class for natural boundary conditions.
 *
 * \tparam ElementType  Element type
 * \tparam DofsPerNode  number degrees of freedom per node
 * \tparam DofOffset    degrees of freedom offset
 *
*******************************************************************************/
template<
  typename ElementType,
  Plato::OrdinalType NumDofs=ElementType::mNumSpatialDims,
  Plato::OrdinalType DofsPerNode=NumDofs,
  Plato::OrdinalType DofOffset=0>
class NaturalBC
{
    const std::string mName; /*!< user-defined load sublist name */
    Plato::Neumann mType;
    std::string mSidesetName;  /*!< side set name */
    std::unique_ptr<NaturalBCData<NumDofs>> mData;
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aLoadName user-defined name for natural boundary condition sublist
     * \param [in] aSubList  natural boundary condition input parameter sublist
    *******************************************************************************/
    NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>(const std::string & aLoadName, const Teuchos::ParameterList& aSublist) :
        mName(aLoadName),
        mType(naturalBoundaryCondition(getStringDataAndAffirmExists("Type", aSublist, mName))),
        mSidesetName(getStringDataAndAffirmExists("Sides", aSublist, mName)),
        mData(makeNaturalBCData<NumDofs>(aSublist))
    {
    }

    /***************************************************************************//**
     * \brief Get the contribution to the assembled forcing vector.
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
     * The boundary terms are integrated on the parameterized surface, \f$\phi(\xi,\psi)\f$, according to:
     *  \f{eqnarray*}{
     *    \phi(\xi,\psi)=
     *       \left\{
     *        \begin{array}{ccc}
     *          N_I\left(\xi,\psi\right) x_I &
     *          N_I\left(\xi,\psi\right) y_I &
     *          N_I\left(\xi,\psi\right) z_I
     *        \end{array}
     *       \right\} \\
     *     f^{el}_{Ii} = \int_{\partial\Omega_{\xi}} N_I\left(\xi,\psi\right) t_i
     *          \left|\left|
     *            \frac{\partial\phi}{\partial\xi} \times \frac{\partial\phi}{\partial\psi}
     *          \right|\right| d\xi d\psi
     * \f}
    *******************************************************************************/
    template<typename StateScalarType,
             typename ControlScalarType,
             typename ConfigScalarType,
             typename ResultScalarType>
    void get(const Plato::SpatialModel&,
             const Plato::ScalarMultiVectorT<  StateScalarType>&,
             const Plato::ScalarMultiVectorT<ControlScalarType>&,
             const Plato::ScalarArray3DT    < ConfigScalarType>&,
             const Plato::ScalarMultiVectorT< ResultScalarType>&,
             Plato::Scalar aScale,
             Plato::Scalar aCurrentTime);

    /***************************************************************************//**
     * \brief Return natural boundary condition sublist name
     * \return sublist name
    *******************************************************************************/
    const std::string& getSubListName() const { return mName; }

    /***************************************************************************//**
     * \brief Return side set name for this natural boundary condition
     * \return side set name
    *******************************************************************************/
    const std::string& getSideSetName() const { return mSidesetName; }

    /***************************************************************************//**
     * \brief Return natural boundary condition type 
     * \return natural boundary condition type
    *******************************************************************************/
    Neumann getType() const { return mType; }

    const NaturalBCData<NumDofs>& getNaturalBCData() const { return *mData; }
    
private:

    void setBCType();
    void setSidesetName();

}; // class NaturalBC

/***************************************************************************//**
 * \brief NaturalBC::get function definition
*******************************************************************************/
template<typename ElementType, Plato::OrdinalType NumDofs, Plato::OrdinalType DofsPerNode, Plato::OrdinalType DofOffset>
template<typename StateScalarType,
         typename ControlScalarType,
         typename ConfigScalarType,
         typename ResultScalarType>
void NaturalBC<ElementType, NumDofs, DofsPerNode, DofOffset>::get(
    const Plato::SpatialModel                          & aSpatialModel,
    const Plato::ScalarMultiVectorT<  StateScalarType> & aState,
    const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
    const Plato::ScalarArray3DT    < ConfigScalarType> & aConfig,
    const Plato::ScalarMultiVectorT< ResultScalarType> & aResult,
          Plato::Scalar aScale,
          Plato::Scalar aCurrentTime
)
{
    switch(mType)
    {
        case Plato::Neumann::UNIFORM_LOAD:
        case Plato::Neumann::VARIABLE_LOAD:
        {
            Plato::SurfaceLoadIntegral<ElementType, NumDofs, DofsPerNode, DofOffset> tSurfaceLoad(mSidesetName, aCurrentTime, mData->clone());
            tSurfaceLoad(aSpatialModel, aState, aControl, aConfig, aResult, aScale);
            break;
        }

        case Plato::Neumann::UNIFORM_PRESSURE:
        case Plato::Neumann::VARIABLE_PRESSURE:
        {
            Plato::SurfacePressureIntegral<ElementType, NumDofs, DofsPerNode, DofOffset> tSurfacePress(mSidesetName, aCurrentTime, mData->clone());
            tSurfacePress(aSpatialModel, aState, aControl, aConfig, aResult, aScale);
            break;
        }
        case Plato::Neumann::STEFAN_BOLTZMANN:
        {
            Plato::StefanBoltzmann<ElementType, DofsPerNode, DofOffset> tStefanBoltzmann(mSidesetName);
            tStefanBoltzmann(aSpatialModel, aState, aControl, aConfig, aResult, aScale);
            break;
        }
        default:
        {
            ANALYZE_THROWERR("Natural Boundary Condition: Natural Boundary Condition Type is NOT supported.")
        }
    }
}

}
// namespace Plato
