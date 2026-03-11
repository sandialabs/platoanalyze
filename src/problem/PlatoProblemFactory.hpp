/*
 * PlatoProblemFactory.hpp
 *
 *  Created on: Apr 19, 2018
 */

#ifndef PLATOPROBLEMFACTORY_HPP_
#define PLATOPROBLEMFACTORY_HPP_

#include <Teuchos_ParameterList.hpp>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "element/Hex27.hpp"
#include "element/Hex8.hpp"
#include "element/Quad4.hpp"
#include "element/Tet10.hpp"
#include "element/Tet4.hpp"
#include "mesh/PlatoMesh.hpp"
#include "problem/Mechanics.hpp"
#include "problem/Thermal.hpp"
#include "problem/Thermomechanics.hpp"
#include "utilities/AnalyzeMacros.hpp"
#include "utilities/ParallelComm.hpp"

#ifdef PLATO_ELLIPTIC
#include "problem/elliptic/Problem.hpp"
#include "problem/elliptic/finite_deformation_mechanics/FiniteDeformationMechanics.hpp"
#include "problem/elliptic/finite_deformation_mechanics/Problem.hpp"
#endif

#ifdef PLATO_PARABOLIC
#include "problem/parabolic/Problem.hpp"
#endif

#ifdef PLATO_HYPERBOLIC
#include "problem/hyperbolic/Mechanics.hpp"
#include "problem/hyperbolic/Problem.hpp"

#endif
#include "problem/helmholtz/AdjointProblem.hpp"
#include "problem/helmholtz/Helmholtz.hpp"
#include "problem/helmholtz/Problem.hpp"

// #include "StructuralDynamicsProblem.hpp"

namespace Plato
{

/******************************************************************************/
/**
 * \brief Check if input PDE type is supported by Analyze.
 * \param [in] aPlatoProb input xml metadata
 * \returns return lowercase pde type
 **********************************************************************************/
inline std::string is_pde_constraint_supported(Teuchos::ParameterList& aPlatoProb)
{
    if (aPlatoProb.isParameter("PDE Constraint") == false)
    {
        ANALYZE_THROWERR("Parameter 'PDE Constraint' is not defined in 'Plato Problem' parameter list.")
    }
    auto tPDE = aPlatoProb.get<std::string>("PDE Constraint");
    auto tLowerPDE = Plato::tolower(tPDE);
    return tLowerPDE;
}
// function is_pde_constraint_supported

template <template <typename> typename ProblemT, template <typename> typename PhysicsT>
inline std::shared_ptr<Plato::AbstractProblem> makeProblem(Plato::Mesh aMesh,
                                                           Teuchos::ParameterList& aPlatoProb,
                                                           Comm::Machine aMachine)
{
    auto tElementType = aMesh->ElementType();
    if (Plato::tolower(tElementType) == "tet10" || Plato::tolower(tElementType) == "tetra10")
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Tet10>>>(aMesh, aPlatoProb, aMachine);
    }
    if (Plato::tolower(tElementType) == "tetra" || Plato::tolower(tElementType) == "tetra4" ||
        Plato::tolower(tElementType) == "tet4" || Plato::tolower(tElementType) == "tet")
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Tet4>>>(aMesh, aPlatoProb, aMachine);
    }
    if (Plato::tolower(tElementType) == "tri" || Plato::tolower(tElementType) == "tri3")
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Tri3>>>(aMesh, aPlatoProb, aMachine);
    }
    if (Plato::tolower(tElementType) == "hex8" || Plato::tolower(tElementType) == "hexa8" ||
        Plato::tolower(tElementType) == "hex")
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Hex8>>>(aMesh, aPlatoProb, aMachine);
    }
    if (Plato::tolower(tElementType) == "hex27" || Plato::tolower(tElementType) == "hexa27")
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Hex27>>>(aMesh, aPlatoProb, aMachine);
    }
    if (Plato::tolower(tElementType) == "quad4")
    {
        return std::make_shared<ProblemT<PhysicsT<Plato::Quad4>>>(aMesh, aPlatoProb, aMachine);
    }
    {
        std::stringstream ss;
        ss << "Unknown mesh type: " << tElementType;
        ANALYZE_THROWERR(ss.str());
    }
}

/******************************************************************************/
/**
 * \brief Create mechanical problem.
 * \param [in] aMesh        plato abstract mesh
 * \param [in] aPlatoProb   input xml metadata
 * \param [in] aMachine     mpi communicator interface
 * \returns shared pointer to abstract problem of type mechanical
 **********************************************************************************/
inline std::shared_ptr<Plato::AbstractProblem> create_mechanical_problem(Plato::Mesh aMesh,
                                                                         Teuchos::ParameterList& aPlatoProb,
                                                                         Comm::Machine aMachine)
{
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if (tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Mechanics>(aMesh, aPlatoProb, aMachine);
    }

#endif
#ifdef PLATO_HYPERBOLIC
    if (tLowerPDE == "hyperbolic")
    {
        return makeProblem<Plato::Hyperbolic::Problem, Plato::Hyperbolic::Mechanics>(aMesh, aPlatoProb, aMachine);
    }
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
}
// function create_mechanical_problem

/******************************************************************************/
/**
 * \brief Create a abstract problem of type thermal.
 * \param [in] aMesh      mesh metadata
 * \param [in] aPlatoProb input xml metadata
 * \param [in] aMachine   mpi communicator interface
 * \returns shared pointer to abstract problem of type thermal
 **********************************************************************************/
inline std::shared_ptr<Plato::AbstractProblem> create_thermal_problem(Plato::Mesh aMesh,
                                                                      Teuchos::ParameterList& aPlatoProb,
                                                                      Comm::Machine aMachine)
{
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_PARABOLIC
    if (tLowerPDE == "parabolic")
    {
        return makeProblem<Plato::Parabolic::Problem, Plato::Thermal>(aMesh, aPlatoProb, aMachine);
    }
#endif
#ifdef PLATO_ELLIPTIC
    if (tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Thermal>(aMesh, aPlatoProb, aMachine);
    }
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
}
// function create_thermal_problem

/******************************************************************************/
/**
 * \brief Create a abstract problem of type thermomechanical.
 * \param [in] aMesh        mesh metadata
 * \param [in] aPlatoProb input xml metadata
 * \param [in] aMachine     mpi communicator interface
 * \returns shared pointer to abstract problem of type thermomechanical
 **********************************************************************************/
inline std::shared_ptr<Plato::AbstractProblem> create_thermomechanical_problem(Plato::Mesh aMesh,
                                                                               Teuchos::ParameterList& aPlatoProb,
                                                                               Comm::Machine aMachine)
{
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_PARABOLIC
    if (tLowerPDE == "parabolic")
    {
        return makeProblem<Plato::Parabolic::Problem, Plato::Thermomechanics>(aMesh, aPlatoProb, aMachine);
    }
#endif
#ifdef PLATO_ELLIPTIC
    if (tLowerPDE == "elliptic")
    {
        return makeProblem<Plato::Elliptic::Problem, Plato::Thermomechanics>(aMesh, aPlatoProb, aMachine);
    }
#endif
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE + "' is not supported.");
    }
}
// function create_thermomechanical_problem

/// @brief Create finite deformation* mechanics problem.
/// @param[in] aMesh plato abstract mesh
/// @param[in] aPlatoProb input xml metadata
/// @param[in] aMachine mpi communicator interface
/// @returns shared pointer to abstract problem of type mechanical finite deformation mechanics
inline std::shared_ptr<Plato::AbstractProblem> create_finite_deformation_mechanics_problem(
    Plato::Mesh aMesh, Teuchos::ParameterList& aPlatoProb, Comm::Machine aMachine)
{
    namespace pefdm = plato::elliptic::finite_deformation_mechanics;
    auto tLowerPDE = Plato::is_pde_constraint_supported(aPlatoProb);

#ifdef PLATO_ELLIPTIC
    if (tLowerPDE == "elliptic")
    {
        return makeProblem<pefdm::Problem, pefdm::FiniteDeformationMechanics>(aMesh, aPlatoProb, aMachine);
    }
    else
    {
        ANALYZE_THROWERR(std::string("'PDE Constraint' of type '") + tLowerPDE +
                         "' is not supported for finite deformation mechanics. Only Elliptic is currently supported.");
    }
#endif
}
// function create_finite_deformation_mechanics_problem

/******************************************************************************/
/**
 * \brief This class is responsible for the creation of a Plato problem, which enables
 * finite element simulations of multiphysics problem.
 **********************************************************************************/
class ProblemFactory
{
   public:
    /******************************************************************************/
    /**
     * \brief Returns a shared pointer to a PLATO problem
     * \param [in] aMesh        abstract mesh
     * \param [in] aInputParams xml metadata
     * \param [in] aMachine     mpi communicator interface
     * \returns shared pointer to a PLATO problem
     **********************************************************************************/
    std::shared_ptr<Plato::AbstractProblem> create(Plato::Mesh aMesh,
                                                   Teuchos::ParameterList& aInputParams,
                                                   Comm::Machine aMachine)
    {
        auto tInputData = aInputParams.sublist("Plato Problem");
        const auto& tPhysics = tInputData.get<std::string>("Physics");
        const auto tLowerPhysics = Plato::tolower(tPhysics);
        if (tLowerPhysics == "mechanical")
        {
            return (Plato::create_mechanical_problem(aMesh, tInputData, aMachine));
        }
        if (tLowerPhysics == "finite deformation mechanics")
        {
            return (Plato::create_finite_deformation_mechanics_problem(aMesh, tInputData, aMachine));
        }
        if (tLowerPhysics == "thermal")
        {
            return (Plato::create_thermal_problem(aMesh, tInputData, aMachine));
        }
        if (tLowerPhysics == "thermomechanical")
        {
            return (Plato::create_thermomechanical_problem(aMesh, tInputData, aMachine));
        }

        if (tLowerPhysics == "helmholtz filter")
        {
            return makeProblem<Plato::Helmholtz::Problem, Plato::HelmholtzFilter>(aMesh, tInputData, aMachine);
        }
        if (tLowerPhysics == "adjoint helmholtz filter")
        {
            return makeProblem<Plato::Helmholtz::AdjointProblem, Plato::HelmholtzFilter>(aMesh, tInputData, aMachine);
        }

        {
            ANALYZE_THROWERR(std::string("'Physics' of type ") + tLowerPhysics + "' is not supported.");
        }
        return nullptr;
    }
};
// class ProblemFactory

}  // namespace Plato
// namespace Plato

#endif /* PLATOPROBLEMFACTORY_HPP_ */
