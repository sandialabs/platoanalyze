/*
 * FluidsCriterionFactory.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <memory>

#include <Teuchos_ParameterList.hpp>

#include "SpatialModel.hpp"
#include "hyperbolic/fluids/FluidsCriterionBase.hpp"

namespace Plato
{

namespace Fluids
{

/**************************************************************************//**
* \struct CriterionFactory
*
* \brief Responsible for the construction of Plato criteria.
******************************************************************************/
template<typename PhysicsT>
class CriterionFactory
{
public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    CriterionFactory () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~CriterionFactory() {}

    /******************************************************************************//**
     * \brief Create criterion interface.
     * \param [in] aModel   computational model metadata
     * \param [in] aDataMap output database
     * \param [in] aInputs  input file metadata
     * \param [in] aTag    scalar function tag
     **********************************************************************************/
    std::shared_ptr<Plato::Fluids::CriterionBase>
    createCriterion
    (const Plato::SpatialModel    & aModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aInputs,
           std::string            & aTag);
};
// class CriterionFactory

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<3>>;
#endif
