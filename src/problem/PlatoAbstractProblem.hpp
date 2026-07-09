/*
 * PlatoAbstractProblem.hpp
 *
 *  Created on: April 19, 2018
 */

#ifndef PLATOABSTRACTPROBLEM_HPP_
#define PLATOABSTRACTPROBLEM_HPP_

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCPDecl.hpp>
#include <filesystem>

#include "domain/Solutions.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"

namespace Plato
{

/// @brief Given a parameter list @a aInputs and a mesh @a aMesh, create a data map that acts like a global scratch
/// space for all problems.
[[nodiscard]] auto make_data_map(const Teuchos::ParameterList& aInputs, const Plato::Mesh& aMesh) -> Plato::DataMap;

/******************************************************************************/
/**
 * \brief Abstract interface for a PLATO problem
 **********************************************************************************/
class AbstractProblem
{
   public:
    /******************************************************************************/
    /**
     * \brief PLATO abstract problem destructor
     **********************************************************************************/
    virtual ~AbstractProblem() = default;

    /******************************************************************************/
    /**
     * \brief PLATO abstract problem constructor
     **********************************************************************************/
    AbstractProblem() {}

    AbstractProblem(Plato::DataMap aDataMap) : mDataMap(std::move(aDataMap)) {}

    /******************************************************************************/
    /**
     * \brief Output solution to visualization file.
     * \param [in] aFilename output file name
     **********************************************************************************/
    virtual void output(const std::filesystem::path& aFilename) const = 0;

    /******************************************************************************/
    /**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D container of control variables
     * \param [in] aSolution solution database
     **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarVector& aControl, const Plato::Solutions& aSolution) = 0;

    /******************************************************************************/
    /**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return 2D view of state variables
     **********************************************************************************/
    virtual Plato::Solutions solution(const Plato::ScalarVector& aControl) = 0;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return criterion function value
     **********************************************************************************/
    virtual Plato::Scalar criterionValue(const Plato::ScalarVector& aControl,
                                         const Plato::Solutions& aSolution,
                                         const std::string& aName) = 0;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
     **********************************************************************************/
    virtual Plato::ScalarVector criterionGradient(const Plato::ScalarVector& aControl,
                                                  const Plato::Solutions& aSolution,
                                                  const std::string& aName) = 0;

    /******************************************************************************/
    /**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt configuration variables
     **********************************************************************************/
    virtual Plato::ScalarVector criterionGradientX(const Plato::ScalarVector& aControl,
                                                   const Plato::Solutions& aSolution,
                                                   const std::string& aName) = 0;

    /******************************************************************************/
    /**
     * \fn const Plato::DataMap getDataMap
     * \brief Return constant reference to Plato output database.
     * \return constant reference to Plato output database
     **********************************************************************************/
    Plato::DataMap mDataMap;
    const decltype(mDataMap)& getDataMap() const { return mDataMap; }

    /******************************************************************************/
    /**
     * \brief Return solution database.
     * \return solution database
     **********************************************************************************/
    virtual Plato::Solutions getSolution() const = 0;
};
// end class AbstractProblem

}  // end namespace Plato

#endif /* PLATOABSTRACTPROBLEM_HPP_ */
