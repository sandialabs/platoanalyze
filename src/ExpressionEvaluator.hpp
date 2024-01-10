/*
  Original code provided by Rhyscitlema
  https://www.rhyscitlema.com/algorithms/expression-parsing-algorithm

  Modified to accept additional operations such log, pow, sqrt, abs
  etc. Also added the abilitiy to parse variables.
*/

#ifndef EXPRESSION_EVALUATOR_HPP
#define EXPRESSION_EVALUATOR_HPP

#include "AnalyzeMacros.hpp"
#include "PlatoTypes.hpp"
#include "ParseTools.hpp"

#include <Sacado.hpp>

#include <cmath>
#include <deque>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace Plato
{

// Using std classes as member variables within the class but not
// within any Kokkos functions is allowed but can cause an erroneous
// warning with some compilers (There is no warning with NVHPC but
// there is with nvcc). Such warning are unknown with HIP/SYCL.

// Using std classes allows for dynamic thus more efficent memory
// allocation. Conversely not using std classes forces one to use
// memory inefficently via static length arrays which places artifical
// limits on the expressions.

//#define USE_STD_CLASSES

#if defined(USE_STD_CLASSES)
  #pragma nv_diag_suppress 20011
#else
  #define MAX_NUM_NODES 256  // Maximum number of nodes in an expression
#endif

// To add flexibility to the expressions it is possible to assign
// variables on a per thread basis (aka hetrogenous variable
// assignment). To do this assignment requires using a Kokkos view of
// views. Typically such usage is discouraged. However, in a code
// review with the Kokkos team, Christian Trott noted that this
// assignment was a legitimate usage in order to achieve dynamic
// allocation. The code was originally developed to support such
// assignments.

// The assignment of variables on a per thread basis can be done
// outside and inside of a Kokkos parallel_for.

// Conversely, using a Plato::Array/Plato::Martix to store the views
// works does not allow for hetrogenous variable assignments because:
//
//   The Plato::Array/Plato::Marix can not be dynamically allocated
//   and forces one to use memory inefficently via static length
//   arrays which places artifical limits on the expressions.
//
//   Array/Marix assignments are not const and thus cannot be called
//   within a parallel_for.

//#define USE_KOKKOS_VIEW_OF_VIEWS

#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
  #define EXPRESSION_EVALUATOR_USE_PER_THREAD_VAR_ASSIGNMENT
  #define EE_KOKKOS_INLINE_FUNCTION KOKKOS_INLINE_FUNCTION
  #define EE_CONST const
#else
  #define EE_KOKKOS_INLINE_FUNCTION
  #define EE_CONST

  #if !defined(MAX_NUM_NODES)
    #define MAX_NUM_NODES 256  // Maximum number of nodes in an expression
  #endif
#endif

// Defines used for how the expression is evaluated. Currently,
// recursion is not possible on GPUs so post order is used. But on
// CPUs recursion is possible and is good for debugging issues.
#define USE_POST_ORDER true
#define USE_RECURSION false

// ************************************************************************* //
// Note: It is assumed that the ResultType, StateType, and VectorType
// are of type Kokkos::View so to properly handle the view of views
// de-referencing correctly. See clear_storage().
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
class ExpressionEvaluator
{
public:
  ExpressionEvaluator();

  void initialize( Kokkos::View< VariableMap *, Plato::UVMSpace > & aVarMaps,
                   const Teuchos::ParameterList & aInputParams,
                   const std::vector< std::string > aParamLabels,
                   const Plato::OrdinalType nThreads,
                   const Plato::OrdinalType nValues );

  void    parse_expression( const char* expression );
  bool    valid_expression( const bool checkVariables = false ) const;

  // Built in parallel_for
  void evaluate_expression( ResultType const & result ) const;

  // Can be called within a parallel_for for direct thread control.
  KOKKOS_INLINE_FUNCTION
  void evaluate_expression( const Plato::OrdinalType thread,
                                  ResultType const & result ) const;
  void   delete_expression();
  void    print_expression(       std::ostream &os,
                            const bool print_val = false ) const;

  // Use std classes for dynamic memory allocation but may cause an
  // erroneous compiler warning.
#if defined(USE_STD_CLASSES)
  const std::vector< std::string > & get_variables() const;
#else
  const char **                      get_variables() const;
#endif

  void     setup_storage( const Plato::OrdinalType nThreads,
                          const Plato::OrdinalType nValues );

  void     clear_storage() const;

  // When using a Kokkos view of views EE_* are defined otherwise they
  // are blank. Only const methods can be called from within a Kokkos
  // parallel_for.
  EE_KOKKOS_INLINE_FUNCTION
  void     set_variable ( const char *, const ScalarType & value,
                          const Plato::OrdinalType thread = -1) EE_CONST;
  EE_KOKKOS_INLINE_FUNCTION
  void     set_variable ( const char *, const VectorType & values,
                          const Plato::OrdinalType thread = -1) EE_CONST;
  EE_KOKKOS_INLINE_FUNCTION
  void     set_variable ( const char *, const StateType  & values) EE_CONST;

  void   print_variables( std::ostream & os ) const;

// ************************************************************************* //
  enum struct NodeID  // Node arithmetic operation - note these are
                      // strongly typed (via the struct) because many
                      // of the enums are common words. Strongly
                      // typing them makes them unique.

                      // Note the underscore with SUBTRACTION_
                      // because of a conflict with the SUBTRACTION
                      // defined globally in EGADS: Electronic
                      // Geometry Aircraft Design System.
  {
    EMPTY_NODE,

    OPEN_PARENTHESIS,
    CLOSE_PARENTHESIS,

    OPEN_ABS_BAR,
    CLOSE_ABS_BAR,

    POSITIVE,
    NEGATIVE,

    ADDITION,
    SUBTRACTION_,
    MULTIPLICATION,
    DIVISION,

    EXPONENTIAL,
    LOG,
    POWER,
    SQRT,
    FACTORIAL,

    ABS,

    SIN,
    COS,
    TAN,

    NUMBER,
    VARIABLE
  };

// ************************************************************************* //
  enum struct NodeInfo  // Used for inserting nodes.
  {
    NoInfo,
    SkipClimbUp,
    RightAssociative,
    LeftAssociative,
  };

// ************************************************************************* //
  enum DataSourceID  // Used for quick look up of variable data.
  {
    SCALAR_DATA_SOURCE = 0,
    VECTOR_DATA_SOURCE = 1,
     STATE_DATA_SOURCE = 2,
       MAX_DATA_SOURCE = 10  // Maximum number of variables in an expression
  };

  Plato::OrdinalType mNumDataSources{3};

// ************************************************************************* //
  // Map structure - used with Kokkos so hardwired.
  template< typename KEY_TYPE, typename VALUE_TYPE > struct _Map {
    KEY_TYPE key;
    VALUE_TYPE value;
  };

  template< typename KEY_TYPE, typename VALUE_TYPE >
  using Map = _Map< KEY_TYPE, VALUE_TYPE>;

// ************************************************************************* //
  // Node stucture for expression tree.
  typedef struct _Node {
    NodeID ID{ NodeID::EMPTY_NODE };     // Arithmetic operation

    Plato::OrdinalType precedence{ 0 };  // Precedence in the tree

    Plato::Scalar number{ 0 };           // Scalar value

    char variable[MAX_ARRAY_LENGTH];     // Variable name

    // Normally the tree would utilize pointers to its parent and
    // children nodes. But for execution on the GPU the nodes need to
    // be a defined chunk of memory. As such, instead of pointers, an
    // index is used instead.
    Plato::OrdinalType i_left  { (Plato::OrdinalType) -1 };
    Plato::OrdinalType i_right { (Plato::OrdinalType) -1 };
    Plato::OrdinalType i_parent{ (Plato::OrdinalType) -1 };

    // Index into the memory array for storing the intermediate
    // results.
    Plato::OrdinalType i_memory { (Plato::OrdinalType) -1 };

  } Node;

private:
// ************************************************************************* //
  // All theses methods are support methods.
  void commute_expression();

  void traverse_expression();

  Plato::OrdinalType  insertNode(       Plato::OrdinalType i_current,
                                  const Plato::OrdinalType i_new,
                                  const NodeInfo info );

  NodeID validateNode( const Plato::OrdinalType i_node,
                       const bool checkVariables = false ) const;

  Plato::OrdinalType commuteNode( const Plato::OrdinalType i_node );

  void   traverseNode( const Plato::OrdinalType i_node,
                       const Plato::OrdinalType depth );
public:
  KOKKOS_INLINE_FUNCTION
  void   evaluateNode( const Plato::OrdinalType thread,
                       const Plato::OrdinalType i_node,
                             ResultType const & result ) const;

private:
  void      clearNode( const Plato::OrdinalType i_node );
  void     deleteNode( const Plato::OrdinalType i_node );

  std::string printNodeID( const Plato::OrdinalType i_node,
                           const bool descriptor,
                           const bool print_val = false ) const;

  std::string printNode( const Plato::OrdinalType i_node,
                         const bool print_val = false ) const;

  void printNode(       std::ostream &os,
                  const Plato::OrdinalType i_node,
                  const int indent,
                  const bool print_val = false ) const;

  // ResultType factorial( ResultType n ) const;

// ************************************************************************* //
  // Member data

  // The index to the top level root node in the tree.
  Plato::OrdinalType mTreeRootNode{ (Plato::OrdinalType) -1 };

  // The total number of variable names in the expression.
  Plato::OrdinalType mNumVariables{ 0 };

#if defined(USE_STD_CLASSES)
  std::vector< std::string > mVariableList;
#else
  char mVariableList[MAX_DATA_SOURCE][MAX_ARRAY_LENGTH];
#endif

  // Number of threads to parallize over
  Plato::OrdinalType mNumThreads{ 0 };

  // Number of values to be evaluated.
  Plato::OrdinalType mNumValues{ 0 };

  // Total number of nodes used to construct the expression tree. Some
  // nodes may only be temporarily used. The array of nodes is
  // constructed on the host and used on the device.
  Plato::OrdinalType mNodesUsed{ 0 };
  Kokkos::View< Node *, Plato::UVMSpace > mNodes;

  // Total number of nodes in the expression tree and the post order
  // evaluation.  The array of nodes is constructed on the host and
  // used on the device.
  Plato::OrdinalType mNodeCount{ 0 };
  Kokkos::View< Plato::OrdinalType *, Plato::UVMSpace > mNodeOrder;

  // The maximum number of chunks of temporary memory needed.
  Plato::OrdinalType mNumMemoryChunks{ (Plato::OrdinalType) 0 };

  // A queue to hold indexes to the chunks of temporary memory.
#if defined(USE_STD_CLASSES)
  std::deque<Plato::OrdinalType> mMemQueue;
#else
  Plato::OrdinalType mMemQueueSize{ 0 };
  Plato::OrdinalType mMemQueue[MAX_NUM_NODES];
#endif

  // Array holding the results for the nodes. The space is reused
  // based on the on post order evaluation.
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
  Kokkos::View< ResultType *, Plato::UVMSpace > mResults;
#else
  Plato::Array< MAX_NUM_NODES, ResultType > mResults;
#endif

  // A mapping of the variable names to their coresponding data in the
  // variable arrays - per thread, per variable.
  Kokkos::View< Map< char[MAX_ARRAY_LENGTH], Plato::OrdinalType > **,
                Plato::UVMSpace > mVariableMap;

  // Counts of the data stored in the variable arrays. Counts are need
  // because not all of the storage is used, per thread, per storage
  // (mNumDataSources).
  Kokkos::View< Plato::OrdinalType **, Plato::UVMSpace > mMapCounts;

  // There are three types of variable, scalars are constant and not
  // indexed, vectors are indexed, and state values are indexed by the
  // thread and an index.

  // Note: if the mNumValues is one, vectors are index based on the
  // mNumThreads. mNumValues is greater than one, vectors are index
  // based on the mNumValues. See evaluateNode for the VARIABLE case
  // when the type is VECTOR_DATA_SOURCE.
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
  // Storage for variable data, scalar and vector types are stored on
  // a per thread, per variable basis. Whereas state data is just on
  // per variable basis (as that data is indexed on thread).
  Kokkos::View< ScalarType **, Plato::UVMSpace > mVariableScalarValues;
  Kokkos::View< VectorType **, Plato::UVMSpace > mVariableVectorValues;
  Kokkos::View< StateType   *, Plato::UVMSpace > mVariableStateValues;
#else
  // Storage for variable data, scalar, vector, and state data is just
  // on per variable basis.
  Plato::Array< MAX_DATA_SOURCE, ScalarType > mVariableScalarValues;
  Plato::Array< MAX_DATA_SOURCE, VectorType > mVariableVectorValues;
  Plato::Array< MAX_DATA_SOURCE, StateType  > mVariableStateValues;
#endif

  // Because there are views of views which are reference counted and
  // deleting the parent view DOES NOT de-reference the child views a
  // dummy view with no memory needs to be sent to replace the child
  // so it is de-referenced.
  static ResultType sDummyResult;
  static StateType  sDummyState ;
  static VectorType sDummyVector;

  // Local definition for Kokkos
  KOKKOS_INLINE_FUNCTION int STRCMP (const char *p1, const char *p2) const
  {
    const unsigned char *s1 = (const unsigned char *) p1;
    const unsigned char *s2 = (const unsigned char *) p2;
    unsigned char c1, c2;

    do
    {
      c1 = (unsigned char) *s1++;
      c2 = (unsigned char) *s2++;
      if (c1 == '\0')
        return c1 - c2;
    }
    while (c1 == c2);

    return c1 - c2;
  }

public:
// ************************************************************************* //
  // Debuging hlper methods.
  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( Plato::Scalar val ) const
  // {
  //   printf( "%f ", val );
  // }

  // template< typename T >
  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( T val ) const
  // {
  //   printf( "%f %f ", val.val(), val.dx(0) );
  // }

  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( const char * header, int line, int i, int j,
  //                   Plato::Scalar tVal, Plato::Scalar ttVal ) const
  // {
  //   printf( "scalar %s %d   %i %i   t=%f   tt=%f \n",
  //           header, line, i, j, tVal, ttVal );
  // }

  // template< typename T >
  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( const char * header, int line, int i, int j,
  //                   T tVal, Plato::Scalar ttVal ) const
  // {
  //   printf( "mixed %s %d  %i %i   t=%f %f   tt=%f %f \n",
  //           header, line, i, j, tVal.val(), tVal.dx(0), ttVal );
  // }

  // template< typename T >
  // KOKKOS_INLINE_FUNCTION
  // void localPrintf( const char * header, int line,
  //                   int i, int j, T tVal, T ttVal ) const
  // {
  //   printf( "sacado %s %d  %i %i   t=%f %f   tt=%f %f \n",
  //           header, line, i, j, tVal.val(), tVal.dx(0), ttVal.val(), ttVal.dx(0) );
  // }
};

/******************************************************************************//**
 * \brief Constructor
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
ExpressionEvaluator()
{
}

/******************************************************************************//**
 * \brief getVariableMapping - Parses the expression, gets the
 * expression variables, sets up the variable mapping between the
 * expression and the data, and sets up the storage needed.

 * \param [in] aVarMaps - variable mapping between the expression and the data.
 * \param [in] aInputParams Teuchos parameter list
 * \param [in] nThreads - number of threads being executed
 * \param [in] nValues  - number of values being evaluated
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
initialize( Kokkos::View< VariableMap *, Plato::UVMSpace > & aVarMaps,
            const Teuchos::ParameterList & aInputParams,
            const std::vector< std::string > aParamLabels,
            const Plato::OrdinalType nThreads,
            const Plato::OrdinalType nValues )
{
  Kokkos::Profiling::pushRegion("ExpressionEvaluator::initialize");

  auto tNumParamLabels = aParamLabels.size();

  // No variable mappings initially.
  for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
    aVarMaps(i).key = 0;

  // Get the expression from the parameters.
  std::string tEquationStr = ParseTools::getEquationParam(aInputParams);

  // Parse the expression.
  this->parse_expression(tEquationStr.c_str());

  // For all of the variables found in the expression optionally
  // get their values from the parameter list.

#if defined(USE_STD_CLASSES)
  for( auto const & tVarName : mVariableList )
  {
#else
  for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
  {
    const std::string tVarName(mVariableList[i]);
#endif
    // Here the expression variable is found as a Plato::Scalar
    // so the value comes from the XML and is set directly.
    if( aInputParams.isType<Plato::Scalar>(tVarName) )
    {
      // The value *MUST BE* converted to the ScalarType as it is
      // the type used for all fixed variables.
      ScalarType tVal = aInputParams.get<Plato::Scalar>(tVarName);

      this->set_variable( tVarName.c_str(), tVal );
    }
    // Here the expression variable is found as a string so the
    // values should come from the XML.
    else if( aInputParams.isType<std::string>(tVarName) )
    {
      std::string tVal = aInputParams.get<std::string>(tVarName);

      // These are the names of the parameters passed into the
      // evaluation operator below. If the expression variable
      // "value" matches then the parameter value will be used.
      bool tFound = false;

      for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
      {
        if( tVal == aParamLabels[i] )
        {
          aVarMaps(i).key = 1;
          strcpy( aVarMaps(i).value, tVarName.c_str() );

          tFound = true;
          break;
        }
      }

      if( !tFound )
      {
        std::stringstream errorMsg;
        errorMsg << "Invalid parameter name '" << tVal << "' "
                 << "found for varaible name '" << tVarName << "'. "
                 << "It must be :";

        for( Plato::OrdinalType i=0; i<tNumParamLabels; i++ )
        {
          errorMsg << " '" << aParamLabels[i] << "'";
        }

        errorMsg << ".";

        ANALYZE_THROWERR( errorMsg.str() );
      }
    }
    // Here the expression variable should come from the
    // parameters passed in.
    else
    {
      // These are the names of the parameters passed into the
      // evaluation operator below. If the expression variable
      // name matches then the parameter value will be used.
      bool tFound = false;

      for( Plato::OrdinalType i=0; i<tNumParamLabels; ++i )
      {
        if( tVarName == aParamLabels[i] )
        {
          aVarMaps(i).key = 1;
          strcpy( aVarMaps(i).value, tVarName.c_str() );

          tFound = true;
          break;
        }
      }

      if( !tFound )
      {
        std::stringstream errorMsg;
        errorMsg << "Invalid varaible name '" << tVarName << "'. "
                 << "It must be :";

        for( Plato::OrdinalType i=0; i<tNumParamLabels; i++ )
        {
          errorMsg << " '" << aParamLabels[i] << "'";
        }

        errorMsg << ".";

        ANALYZE_THROWERR(  errorMsg.str() );
      }
    }
  }

  // After the parsing, set up the storage. The sizes must match the
  // input and output data sizes.
  this->setup_storage( nThreads, nValues );

  Kokkos::Profiling::popRegion();
}

/******************************************************************************//**
 * \brief setup_storage - sets up the storage for evaluating the expression
 * \param [in] nThreads - number of threads being executed
 * \param [in] nValues  - number of values being evaluated
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
setup_storage( const Plato::OrdinalType nThreads,
               const Plato::OrdinalType nValues )
{
  Kokkos::Profiling::pushRegion("ExpressionEvaluator::setup_storage");

  mNumThreads = nThreads;
  mNumValues  = nValues;

  // Reference to the results storage.
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
  mResults = Kokkos::View<ResultType *,
                          Plato::UVMSpace>("ExpEval Results", mNumMemoryChunks);
#endif

  // Allocate the actual results storage.
  for( Plato::OrdinalType i=0; i<mNumMemoryChunks; ++i )
  {
    std::stringstream nameStr;
    nameStr << "ExpEval Result " << i;

    mResults[i] = ResultType(nameStr.str(), mNumThreads, mNumValues);
  }

  // Create the variable map array, which maps the variable names to
  // the storage for each thread. It is needed on a thread basis
  // becuase threads operate independently.
  mVariableMap =
    Kokkos::View< Map< char[MAX_ARRAY_LENGTH], Plato::OrdinalType > **,
                  Plato::UVMSpace >("ExpEval VariableMap", mNumThreads, mNumVariables );

  // Fill in the variable names for each thread.
  for( Plato::OrdinalType i=0; i<mNumThreads; ++i )
  {
    for( Plato::OrdinalType j=0; j<mNumVariables; ++j )
    {
#if defined(USE_STD_CLASSES)
      strcpy( mVariableMap(i,j).key, mVariableList[j].c_str() );
#else
      strcpy( mVariableMap(i,j).key, mVariableList[j] );
#endif
      mVariableMap(i,j).value = (Plato::OrdinalType) -1;
    }
  }

  // Counts of the variables stored in each of the maps. These counts
  // are on a thread basis. It is needed on a thread basis becuase
  // threads operate independently.
  mMapCounts =
    Kokkos::View< Plato::OrdinalType **, Plato::UVMSpace >("ExpEval MapCounts", mNumThreads, mNumDataSources);

  for( Plato::OrdinalType i=0; i<mNumThreads; ++i )
  {
    for( Plato::OrdinalType j=0; j<mNumDataSources; ++j )
    {
      mMapCounts(i,j) = 0;
    }
  }

  // Storage of the variable data on a thread basis. The scalars are
  // assumed to be a single value, vectors are a 1D vector, while
  // state variables are assumed to be 2D arrays.
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
  mVariableScalarValues =
    Kokkos::View< ScalarType **,
                  Plato::UVMSpace >("ExpEval Scalar Values", mNumThreads, mNumVariables);
  mVariableVectorValues =
    Kokkos::View< VectorType **,
                  Plato::UVMSpace >("ExpEval Vector Values", mNumThreads, mNumVariables);
  mVariableStateValues =
    Kokkos::View<  StateType *,
                  Plato::UVMSpace >("ExpEval State Values", mNumVariables);
#endif

  Kokkos::Profiling::popRegion();
}

/******************************************************************************//**
 * \brief clear_storage - clear the storage for evaluating the expression
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
clear_storage() const
{
  Kokkos::Profiling::pushRegion("ExpressionEvaluator::clear_storage");

  // Fence before deallocation on host, to make sure that the device
  // kernel is done first.
  Kokkos::fence();

#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
  // Because views of views are used and views are reference counted,
  // deleting the parent view DOES NOT de-reference the child views. A
  // such a dummy view with no memory is used to replace the child so
  // it is de-referenced.

  // Clear the results storage.
  for( Plato::OrdinalType i=0; i<mNumMemoryChunks; ++i )
  {
    mResults[i] = sDummyResult;
  }

  // Clear the state view and the vector which could be any type.
  for( Plato::OrdinalType j=0; j<mNumVariables; ++j )
  {
    mVariableStateValues(j) = sDummyState;

    for( Plato::OrdinalType i=0; i<mNumThreads; ++i )
    {
      mVariableVectorValues(i,j) = sDummyVector;
    }
  }
#endif

  Kokkos::Profiling::popRegion();
}

/******************************************************************************//**
 * \brief get_variables - returns the variables found in the expression.
 * \return std::string - the variable names
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
#if defined(USE_STD_CLASSES)
const std::vector< std::string > &
#else
const char **
#endif
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
get_variables() const
{
  return mVariableList;
}

/******************************************************************************//**
 * \brief set_variable - Sets auxillary variables that are indexed.
                          and may change depending on the thread.
 * \param [in] varName - the variable name
 * \param [in] values  - the variable values
 * \param [in] thread  - the thread being executed
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
EE_KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
set_variable( const char * varName,
              const ScalarType & value,
              const Plato::OrdinalType thread ) EE_CONST
{
  Plato::OrdinalType start, end;

  if( mNumThreads == 0 )
  {
    GPU_WARNING("Invalid call to set_variable - "
                "setup_storage has not been called. "
                "The number of threads has not been set.\n\n");
  }


  // If the default set the value for all threads.
  if( thread == (Plato::OrdinalType) -1 )
  {
    start = 0;
    end = mNumThreads;
  }
  else
  {
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
  // Otherwise set just for this thread.
    start = thread;
    end = thread + 1;
#else
    GPU_WARNING("Invalid call to set_variable - "
                "Can not set the variable on a per thread basis. "
                "Must be recompiled with "
                "'#define USE_KOKKOS_VIEW_OF_VIEWS`.\n\n");
#endif
  }

  bool found = false;

  for( Plato::OrdinalType t=start; t<end; ++t)
  {
    for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
    {
      if( STRCMP( mVariableMap(t, i).key, varName ) == 0 )
      {
        Plato::OrdinalType index = 0;

        if( mVariableMap(t, i).value != (Plato::OrdinalType) -1 )
        {
          // Value exists so replace it.
          index = mVariableMap(t, i).value % MAX_DATA_SOURCE;
        }
        else
        {
          // Value does not exists so add it.
          index = mMapCounts(t, SCALAR_DATA_SOURCE)++;

          // The index gives the index into the storage container. There
          // are three. The *_DATA_SOURCE * MAX_DATA_SOURCE gives the
          // index as to which of the three storage containers is being
          // used. The later makes for easy lookup when evaluating.
          mVariableMap(t, i).value = SCALAR_DATA_SOURCE * MAX_DATA_SOURCE + index;
        }

#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
        mVariableScalarValues(t, index) = value;
#else
        mVariableScalarValues(index) = value;
#endif
        found = true;

        break;
      }
    }
  }

  if(!found)
  {
    GPU_WARNING("Invalid call to set_variable - ");
    printf("The variable '%s' is not part of the expression.\n\n", varName );
  }
}

/******************************************************************************//**
 * \brief set_variable - Sets auxillary variables that are indexed.
                          and may change depending on the thread.
 * \param [in] varName - the variable name
 * \param [in] values  - the variable values
 * \param [in] thread  - the thread being executed
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
EE_KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
set_variable( const char * varName,
              const VectorType & values,
              const Plato::OrdinalType thread ) EE_CONST
{
  Plato::OrdinalType start, end;

  if( mNumThreads == 0 )
  {
    GPU_WARNING("Invalid call to set_variable - "
                "setup_storage has not been called. "
                "The number of threads has not been set.\n\n" );
  }

  // If the default set the value for all threads.
  if( thread == (Plato::OrdinalType) -1 )
  {
    start = 0;
    end = mNumThreads;
  }
  // Otherwise set just for this thread.
  else
  {
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
    if( mNumValues == 1 )
    {
      GPU_WARNING("Invalid call to set_variable - ");
      printf("The vector, '%s' "
             "will be parallelized over all threads but "
             "requesting the vector be used on a per thread basis. "
             "To remove this warning do not pass a thread index.\n\n",
             varName);
    }
    else
    {
      start = thread;
      end = thread + 1;
    }
#else
    GPU_WARNING("Invalid call to set_variable - "
                "Can not set the variable on a per thread basis. "
                "Must be recompiled with "
                "'#define USE_KOKKOS_VIEW_OF_VIEWS`.\n\n");
#endif
  }

  // When the number of values is one the vector will be indexed
  // based on the thread.
  if( mNumValues == 1 && values.extent(0) != mNumThreads )
  {
    GPU_WARNING("Invalid call to set_variable - ");
    printf("Parallelizing over the threads but "
           "the vector, '%s' has %d values. "
           "The number of expected values is %d.\n\n",
           varName, (int) values.extent(0), (int) mNumThreads);
  }
  // When the number of values is greater than one the vector
  // will be indexed over the number values.
  else if( mNumValues > 1 && values.extent(0) != mNumValues )
  {
    GPU_WARNING("Invalid call to set_variable - ");
    printf("Indexing over the number of values but "
           "the vector, '%s' has %d values. "
           "The number of expected values is %d.\n\n",
           varName, (int) values.extent(0), (int) mNumValues);
  }

  bool found = false;

  for( Plato::OrdinalType t=start; t<end; ++t)
  {
    for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
    {
      if( STRCMP( mVariableMap(t, i).key, varName ) == 0 )
      {
        Plato::OrdinalType index = 0;

        if( mVariableMap(t, i).value != (Plato::OrdinalType) -1 )
        {
          // Value exists so replace it.
          index = mVariableMap(t, i).value % MAX_DATA_SOURCE;
        }
        else
        {
          // Value does not exists so add it.
          index = mMapCounts(t, VECTOR_DATA_SOURCE)++;

          // The index gives the index into the storage container. There
          // are three. The *_DATA_SOURCE * MAX_DATA_SOURCE gives the
          // index as to which of the three storage containers is being
          // used. The later makes for easy lookup when evaluating.
          mVariableMap(t, i).value = VECTOR_DATA_SOURCE * MAX_DATA_SOURCE + index;
        }

#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
        mVariableVectorValues(t, index) = values;
#else
        mVariableVectorValues(index) = values;
#endif
        found = true;

        break;
      }
    }
  }

  if(!found)
  {
    GPU_WARNING("Invalid call to set_variable - ");
    printf("The variable '%s' is not part of the expression.\n\n", varName );
  }
}

/******************************************************************************//**
 * \brief set_variable - Sets input variables that are assumed to be multiple
                         vectors the indexing is across all threads.
 * \param [in] varName - the variable name
 * \param [in] values  - the variable values
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
EE_KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
set_variable( const char * varName,
              const StateType & values ) EE_CONST
{
  if( mNumThreads == 0 )
  {
    GPU_WARNING("Invalid call to set_variable - "
                "setup_storage has not been called."
                "The number of threads has not been set.\n\n");
  }

  if( values.extent(0) != mNumThreads ||
      values.extent(1) != mNumValues )
  {
    GPU_WARNING("Invalid call to set_variable - ");

    if( values.extent(0) != mNumThreads )
      printf("The vector, '%s' first extent has %d values. "
             "The number of expected threads is %d.\n\n",
             varName, (int) values.extent(0), mNumThreads);

    if( values.extent(1) != mNumValues )
      printf("The vector, '%s' second extent has %d values. "
             "The number of expected values is %d.\n\n",
             varName, (int) values.extent(1), mNumValues);
  }

  bool found = false;

  // Even though there is only a single input across all threads set
  // up the map for each thread so that the map can be used regardless
  // of which thread is being processed. This is opposed to an unique
  // input for each thread which is the case for a scalar and vector
  // values above.
  for( Plato::OrdinalType t = 0; t<mNumThreads; ++t)
  {
    for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
    {
      if( STRCMP( mVariableMap(t, i).key, varName ) == 0 )
      {
        Plato::OrdinalType index = 0;

        if( mVariableMap(t, i).value != (Plato::OrdinalType) -1 )
        {
          // Value exists so replace it.
          index = mVariableMap(t, i).value % MAX_DATA_SOURCE;
        }
        else
        {
          // Value does not exists so add it.
          index = mMapCounts(t, STATE_DATA_SOURCE)++;

          // The index gives the index into the storage container. There
          // are three. The *_DATA_SOURCE * MAX_DATA_SOURCE gives the
          // index as to which of the three storage containers is being
          // used. The later makes for easy lookup when evaluating.
          mVariableMap(t, i).value = STATE_DATA_SOURCE * MAX_DATA_SOURCE + index;
        }

        mVariableStateValues(index) = values;

        found = true;

        break;
      }
    }
  }

  if(!found)
  {
    GPU_WARNING("Invalid call to set_variable - ");
    printf("The variable '%s' is not part of the expression.\n\n", varName );
  }
}


/******************************************************************************//**
 * \brief print_variables - Print method for the variables in the expression.
 * \param [in] os - the output stream
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
print_variables( std::ostream &os ) const
{
  os << "Plato::Scalar variables (index, name, value)" << std::endl;

  Plato::OrdinalType thread = 0;

  for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
  {
    Plato::OrdinalType type  = (Plato::OrdinalType) -1;
    Plato::OrdinalType index = (Plato::OrdinalType) -1;

    if( mVariableMap(thread, i).value != (Plato::OrdinalType) -1 )
    {
      // Decode the value The type indicated the storage
      // container. There are three. The index gives the location into
      // the storage container being used.
      type  = mVariableMap(thread, i).value / MAX_DATA_SOURCE;
      index = mVariableMap(thread, i).value % MAX_DATA_SOURCE;
    }
    else
    {
      std::stringstream errorMsg;
      errorMsg << "Invalid call to printNode - "
               << "can not find values for variable: " << mVariableMap(thread, i).key;
      ANALYZE_THROWERR( errorMsg.str() );
    }

    os << i << "  " << mVariableMap(thread, i).key << "  ";

    // Get the data from the storage container.
    if( type == SCALAR_DATA_SOURCE )
    {
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
      const ScalarType & value = mVariableScalarValues(thread, index);
#else
      const ScalarType & value = mVariableScalarValues(index);
#endif
      os << value << "  ";
    }
    else if( type == VECTOR_DATA_SOURCE )
    {
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
      const VectorType & values = mVariableVectorValues(thread, index);
#else
      const VectorType & values = mVariableVectorValues(index);
#endif
      os << values[0] << "  ";

      if( mNumValues > 1 )
        os << "...  ";
    }
    else if( type == STATE_DATA_SOURCE )
    {
      const StateType & values = mVariableStateValues(index);

      os << values(0,0) << "  ";

      if( mNumValues > 1 )
        os << "...  ";
    }
    else
    {
      std::stringstream errorMsg;
      errorMsg << "Invalid call to print_variables - "
               << "can not find variable: " << mVariableMap(thread, i).key;
      ANALYZE_THROWERR( errorMsg.str() );
    }

    os << std::endl;
  }
}

/******************************************************************************//**
 * \brief valid_expression - Validate the nodes in the tree.
 * \param [in] checkVariables - check whether values have assigned to variables.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
bool
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
valid_expression( const bool checkVariables ) const
{
  // Return false if a non empty node is found.
  return (validateNode( mTreeRootNode, checkVariables ) ==
          ExpressionEvaluator::NodeID::EMPTY_NODE);
}

/******************************************************************************//**
 * \brief evaluate_expression - Evaluate the expression tree.
 * \param [out] result - resulting data.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
evaluate_expression( ResultType const & result ) const
{
    Kokkos::parallel_for("ExpressionEvaluator",
    Kokkos::RangePolicy<>(0, mNumThreads),
      KOKKOS_CLASS_LAMBDA(Plato::OrdinalType aCellOrdinal)
      {
        this->evaluate_expression( aCellOrdinal, result );
      });
}

/******************************************************************************//**
 * \brief evaluate_expression - Evaluate the expression tree.
 * \param [in]  thread - thread being evaluated.
 * \param [out] result - resulting data.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
evaluate_expression( const Plato::OrdinalType thread,
                           ResultType const & result ) const
{
#if USE_POST_ORDER
  // Post order evaluation of the nodes. Note the last node is the top
  // node which is evaluated separtately with the results going into
  // the return results instead of temporary results
  for( Plato::OrdinalType i=0; i<mNodeCount-1; ++i )
  {
    Plato::OrdinalType i_node = mNodeOrder[i];

    evaluateNode( thread, i_node, mResults[ mNodes[i_node].i_memory ] );
  }
#endif

  evaluateNode( thread, mTreeRootNode, result );
}

/******************************************************************************//**
 * \brief delete_expression - Delete the expression tree.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
delete_expression()
{
  deleteNode( mTreeRootNode );

  mTreeRootNode = (Plato::OrdinalType) -1;
}

/******************************************************************************//**
 * \brief print_expression - print the expression tree.
 * \param [in] os - the output stream
 * \param [in] print_val - print variable value(s)
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
print_expression(       std::ostream &os,
                  const bool print_val ) const
{
  printNode( os, mTreeRootNode, 4, print_val );
}

/******************************************************************************//**
 * \brief commute_expression - traverse the expression and commute
 * nodes so to have a left weighted tree which requires less memory
 * when evaluated without recursion.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
commute_expression()
{
  commuteNode( mTreeRootNode );
}

/******************************************************************************//**
 * \brief traverse_expression - traverse the expression to get the
 * total node count and the post order evaluation and the number of
 * chunks of temporary memory required.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
traverse_expression()
{
  // Before traversing the tree commute the expression so to have a
  // left weighted tree which requires less memory when evaluated
  // without recursion.
  commute_expression();

  Kokkos::Profiling::pushRegion("ExpressionEvaluator::traverse_expression");

  // Get the post order node evaluation order and the maximum number
  // of temporary memory chunks needed.
  mNodeOrder = Kokkos::View< Plato::OrdinalType *,
                             Plato::UVMSpace >( "ExpEval NodeOrder", mNodesUsed );

  mNodeCount = 0;

  // Initialize the memory queue which are a series of indexes that
  // point to chunks of memory that is reused. Thus minimizing the
  // amount of temporary memory needed. The maximum number chunks of
  // temporary memory required is the depth of the epresssion tree
  // plus one. It gets filled in as needed.

  // Example if the depth of tree is 3, the post order evaluation is
  // ACBEGFD. Where node A used memory chunk 0, node C used memory
  // chunk 1, the result (node B) goes into memory chunk 2. At which
  // point memory chunks 0 and 1 can be reused by nodes E and G. Their
  // result (node F) goes into memory chunck 3. At which point memory
  // chunck 0 can be reused again by node D. Though in actuality that
  // result goes into the memory given by the user.
  //
  //           (D0)
  //     (B2)        (F3)
  //  (A0)  (C1)  (E0)  (G1)

#if defined(USE_STD_CLASSES)
  mMemQueue.clear();
  mMemQueue.push_back(0);
#else
  mMemQueue[0] = 0;
  mMemQueueSize = 1;
#endif
  traverseNode( mTreeRootNode, 1 );

  // Take a subview so to reduce the memory footprint of the node order.
  mNodeOrder = subview(mNodeOrder, std::make_pair(0, mNodeCount-1));

  // When complete the mMemQueue.size() is the tree depth and there
  // should only be one index missing from the list which would
  // normally be used by the top level node. But it is not used as
  // those results go into the memory provided by the user.

  // The worst case maximum number of chunks of temporary memory
  // needed is the expression tree depth plus one. But depending on
  // the shape of the tree it may not all be needed. So find the
  // maximum index.  The total needed will be the maximum index plus
  // one.
  mNumMemoryChunks = 0;

  // Note do not check the last node which is the top level node and
  // temporary memory is not needed for it.
  for( Plato::OrdinalType i=0; i<mNodeCount-1; ++i)
  {
    if( mNumMemoryChunks < mNodes[mNodeOrder[i]].i_memory+1 )
      mNumMemoryChunks = mNodes[mNodeOrder[i]].i_memory+1;
  }

#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
#else
  if(mNumMemoryChunks >= MAX_NUM_NODES)
  {
    std::stringstream errorMsg;
    errorMsg << "The expression tree hs too many nodes. "
             << "A maximum number of nodes " << MAX_NUM_NODES << " is allowed."
             << "Increase MAX_NUM_NODES to allow for a deeper expression tree.";
    ANALYZE_THROWERR( errorMsg.str() );
  }
#endif

  Kokkos::Profiling::popRegion();
}

/******************************************************************************//**
 * \brief parse_expression - parse the expression.
 * \param [in] expression - the expression for parsing.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
parse_expression( const char* expression )
{
  Kokkos::Profiling::pushRegion("ExpressionEvaluator::parse_expression");

  if( mTreeRootNode != (Plato::OrdinalType) -1 )
  {
    delete_expression();

#if defined(USE_STD_CLASSES)
    mVariableList.clear();
#endif
    mNumVariables = 0;
  }

  if( expression == nullptr )
    return;

  std::stringstream errorMsg;

  // Preserve the incoming expression.
  const char* expPtr = expression;

  // Get the expression length plus a buffer of 16 more to account for
  // implied operations such a multiplication.
  const Plato::OrdinalType expLength = strlen(expPtr) + 16;

  // Create the node array - worst case one for each character in the
  // expression. Will reduce to the actual needed at the end.
  mNodesUsed = 0;
  mNodes = Kokkos::View< Node *, Plato::UVMSpace >( "ExpEval Nodes", expLength );

  // The absolute value can be specified via abs(X) or |X| for the
  // latter, |X| it is necessary to keep track of whether the first or
  // second instance of the bar '|' has been found.
  bool beginABS = false;

  // Initialise the tree with an empty node which will be deleted when
  // the parsing is finished.
  Plato::OrdinalType i_root = mNodesUsed++;

  Plato::OrdinalType i_current  = i_root;
  Plato::OrdinalType i_previous = i_root;

  // For tracking two parameter functions and open operands
  std::vector< int > twoParamFunctionLocations;  // Location in the expression
  std::vector< int > openAbsBarLocations;        // Location in the expression
  std::vector< int > openParenLocations;         // Location in the expression

  // Stacks for tracking functions and operands.
  std::vector< NodeID > twoParamFunctionIDs; // Two parameter functions; 'pow'
  std::vector< int >    numOpenParens;       // Number of open '(' on the stack
  std::vector< int >    numOpenAbsBars;      // Number of open '|' on the stack

  numOpenParens.push_back( 0 );  //  Number of open '(' for the root
  numOpenAbsBars.push_back( 0 ); //  Number of open '|' for the root

  // Parse the expression.
  while( true )
  {
    NodeInfo info = NodeInfo::NoInfo;     // Set the info to the default value

    Plato::OrdinalType i_node = mNodesUsed++;

    char c = *expPtr++; // get latest character of string

    // At the end of input string
    if( c == '\0' )
    {
      mNodesUsed--;
      break;
    }
    // Ingnore spaces, tabs, returns, end of lines.
    else if( c == ' ' || c == '\t' || c == '\r' || c == '\n')
    {
      mNodesUsed--;
      continue;
    }

    // Open parenthesis
    else if( c == '(' )
    {
      // Number of open parenthesises relative to an open absolute value bar.
      ++numOpenParens.back();

      // Location of this open parenthesis for a match close parenthesis.
      openParenLocations.push_back( strlen(expression) - strlen(expPtr) - 1 );

      // Number of open absolute value bars on this level.
      numOpenAbsBars.push_back( 0 );
      beginABS = false;

      mNodes[i_node].ID = NodeID::OPEN_PARENTHESIS;
      mNodes[i_node].precedence = 0;
      info = NodeInfo::SkipClimbUp;
    }
    // Close parenthesis
    else if( c == ')' )
    {
      mNodes[i_node].ID = NodeID::CLOSE_PARENTHESIS;
      mNodes[i_node].precedence = 0;
      info = NodeInfo::RightAssociative;

      // When closing make sure there is a corresponding open parenthesis
      if( numOpenParens.back() == 0 )
      {
        errorMsg << "Invalid expression: "
                 << "Found a close parenthesis ')' without a "
                 << "corresponding open parenthesis '('";
        break;
      }
      // When closing make sure there is not an open absolute value bar
      else if( numOpenAbsBars.back() > 0 )
      {
        errorMsg << "Invalid expression: "
                 << "Found a close parenthesis ')' before "
                 << "a close absolute value bar '|'.";
        break;
      }
      else
      {
        // Number of open parenthesis relative to open absolute value bar.
        --numOpenParens.back();
        // No open absolute value bar.
        numOpenAbsBars.pop_back();
        // Matched an open parenthesis
        openParenLocations.pop_back();

        // Set for an open absolute value bar on previous level.
        beginABS = numOpenAbsBars.back();
      }
    }

    // Absolute value use the same delimiter. So use a flag to keep
    // track of it being the first or the second.
    else if( c == '|' )
    {
      // Open absolute value.
      if( beginABS == false )
      {
        beginABS = true;

        // Number of open absolute value bars relative to an open parenthesis.
        ++numOpenAbsBars.back();

        // Location of this open absolute value bar for a match close
        // absolute value bar.
        openAbsBarLocations.push_back( strlen(expression) - strlen(expPtr) - 1 );

        // Number of open parenthesises on this level.
        numOpenParens.push_back( 0 );

        // Create and ABS mNodes[i_node].
        mNodes[i_node].ID = NodeID::OPEN_ABS_BAR;
        mNodes[i_node].precedence = 1;
        info = NodeInfo::SkipClimbUp;
      }
      // Close absolute value.
      else {
        beginABS = false;

        mNodes[i_node].ID = NodeID::CLOSE_ABS_BAR;
        mNodes[i_node].precedence = 1;
        info = NodeInfo::RightAssociative;

        // When closing make sure there is an open absolute value bar
        if( numOpenAbsBars.back() == 0 )
        {
          errorMsg << "Invalid expression: "
                   << "Found a close absolute value bar '|' without a "
                   << "corresponding open absolute value bar '|'";
          break;
        }
        // When closing make sure there is not an open parenthesis
        else if( numOpenParens.back() > 0 )
        {
          errorMsg << "Invalid expression: "
                   << "Found a closed absolute value bar '|' before "
                   << "a close parenthesis ')'.";
          break;
        }
        else
        {
          // Number of open absolute value bars relative to an open parenthesis.
          --numOpenAbsBars.back();
          // No open parenthesises.
          numOpenParens.pop_back();
          // Matched an open absolute value bar
          openAbsBarLocations.pop_back();
        }
      }
    }
    // Operands
    else if( c == '+' || c == '-' )
    {
      // Distinguish between a plus/mius sign vs making a result
      // positive/negative.
      if(   mNodes[i_previous].ID == NodeID::NUMBER
         || mNodes[i_previous].ID == NodeID::VARIABLE
         || mNodes[i_previous].ID == NodeID::FACTORIAL
         || mNodes[i_previous].ID == NodeID::CLOSE_PARENTHESIS
         || mNodes[i_previous].ID == NodeID::CLOSE_ABS_BAR )
      {
        mNodes[i_node].ID = (c == '+' ? NodeID::ADDITION : NodeID::SUBTRACTION_);
        mNodes[i_node].precedence = 2;
        info = NodeInfo::LeftAssociative;
      }
      else {
        mNodes[i_node].ID = (c == '+' ? NodeID::POSITIVE : NodeID::NEGATIVE);
        mNodes[i_node].precedence = 5;
        info = NodeInfo::SkipClimbUp;
      }
    }

    else if( c == '*' ) {
      mNodes[i_node].ID = NodeID::MULTIPLICATION;
      mNodes[i_node].precedence = 3;
      info = NodeInfo::LeftAssociative;
    }
    else if( c == '/' ) {
      mNodes[i_node].ID = NodeID::DIVISION;
      mNodes[i_node].precedence = 3;
      info = NodeInfo::LeftAssociative;
    }
    else if( c == '^' ) {
      mNodes[i_node].ID = NodeID::POWER;
      mNodes[i_node].precedence = 4;
      info = NodeInfo::RightAssociative;
    }
    // else if( c == '!' ) {
    //   mNodes[i_node].ID = FACTORIAL;
    //   mNodes[i_node].precedence = 6;
    //   info = NodeInfo::LeftAssociative;
    // }

    // Functions
    else if( memcmp(expPtr-1, "sin" , 3) == 0 ) {
      expPtr += 3-1;
      mNodes[i_node].ID = NodeID::SIN;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
    }
    else if( memcmp(expPtr-1, "cos" , 3) == 0 ) {
      expPtr += 3-1;
      mNodes[i_node].ID = NodeID::COS;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
    }
    else if( memcmp(expPtr-1, "tan" , 3) == 0 ) {
      expPtr += 3-1;
      mNodes[i_node].ID = NodeID::TAN;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
    }

    else if( memcmp(expPtr-1, "exp" , 3) == 0 ) {
      expPtr += 3-1;
      mNodes[i_node].ID = NodeID::EXPONENTIAL;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
    }
    else if( memcmp(expPtr-1, "log" , 3) == 0 ) {
      expPtr += 3-1;
      mNodes[i_node].ID = NodeID::LOG;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
    }
    else if( memcmp(expPtr-1, "sqrt", 4) == 0 ) {
      expPtr += 4-1;
      mNodes[i_node].ID = NodeID::SQRT;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
    }
    else if( memcmp(expPtr-1, "abs" , 3) == 0 ) {
      expPtr += 3-1;
      mNodes[i_node].ID = NodeID::ABS;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
    }

    // For two argument functions push the function on to the next ID
    // stack. But do not create a mNodes[i_node]. The node will be created when
    // the corresponding comma is found.
    else if( memcmp(expPtr-1, "pow" , 3) == 0 ) {
      expPtr += 3-1; twoParamFunctionIDs.push_back( NodeID::POWER );
      twoParamFunctionLocations.push_back( strlen(expression) - strlen(expPtr) - 1 );

      continue;
    }

    // When a comma is found pop the last function from the stack and
    // make the corresponding mNodes[i_node].
    else if( c == ',' ) {
      if( twoParamFunctionIDs.size() )
      {
        mNodes[i_node].ID = twoParamFunctionIDs.back();
        mNodes[i_node].precedence = 5;
        info = NodeInfo::LeftAssociative;

        twoParamFunctionIDs.pop_back();
        twoParamFunctionLocations.pop_back();
      }
      else
      {
        errorMsg << "Invalid expression: "
                 << "Found a comma ',' without a corresponding operation.";
        break;
      }
    }
    // Fixed constant
    else if( memcmp(expPtr-1, "PI"  , 2) == 0 ) {
      expPtr += 2-1;
      mNodes[i_node].ID = NodeID::NUMBER;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
      mNodes[i_node].number = M_PI;
    }

    // Number
    else if( '0' <= c && c <= '9' )
    {
      char number[MAX_ARRAY_LENGTH];
      int i = 0;

      while( true )
      {
        number[i++] = c;

        if( i+1 == sizeof(number) )
        {
          number[i] = '\0';
          errorMsg << "Invalid expression: "
                   << "The number '" << number << "' exceeds the "
                   << sizeof(number) << " digit limit.";
          break;
        }

        c = *expPtr;

        // Check the next character to see if it is a digit or dot.
        if( ('0' <= c && c <='9') || c == '.' )
        {
          expPtr++;
        }
        // Check for an exponenet.
        else if( c == 'e' )
        {
          expPtr++;

          // Store the character and immediately read the next character.
          number[i++] = c;

          if( i+1 == sizeof(number) )
          {
            number[i] = '\0';
            errorMsg << "Invalid expression: "
                     << "The number '" << number << "' exceeds the "
                     << sizeof(number) << " digit limit.";
            break;
          }

          c = *expPtr;

          // Check the next character to see if it is a digit or a plus
          // or minus sign.
          if( ('0' <= c && c <='9') || c == '+' || c == '-' )
            expPtr++;
          else
            break;
        }
        else
          break;
      }

      // Get the actual number from the string.
      number[i] = '\0';
      sscanf(number, "%lf", &mNodes[i_node].number);
      mNodes[i_node].ID = NodeID::NUMBER;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;
    }

    // Variable
    else if( ('a' <= c && c <= 'z') ||  // It must start with a lower
             ('A' <= c && c <= 'Z') ||  // or upper case letter
             c == '_' )                 // or an underscore.
    {
      char variable[MAX_ARRAY_LENGTH];
      int i = 0;

      while( true )
      {
        variable[i++] = c;

        if( i+1 == sizeof(variable) )
        {
          variable[i] = '\0';

          errorMsg << "Invalid expression: "
                   << "The variable '" << variable << "' exceeds the "
                   << sizeof(variable) << " character limit.";
          break;
        }

        c = *expPtr;

        // Variables may have letters, numbers, and/or underscores ONLY.
        if( ('a' <= c && c <= 'z') ||
            ('A' <= c && c <= 'Z') ||
            ('0' <= c && c <= '9') ||
            c == '_' )
          expPtr++;
        else
          break;
      }

      // Save the variable name.
      variable[i] = '\0';
      strcpy(mNodes[i_node].variable, variable);
      mNodes[i_node].ID = NodeID::VARIABLE;
      mNodes[i_node].precedence = 7;
      info = NodeInfo::LeftAssociative;

      // Record variable the name so have a unique list.
      bool found = false;

#if defined(USE_STD_CLASSES)
      for( auto const & tVarName : mVariableList )
      {
#else
      for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
      {
        std::string tVarName(mVariableList[i]);
#endif
        if( tVarName == std::string(variable) )
        {
          found = true;
          break;
        }
      }

      if( !found )
      {
        if(mNumVariables == MAX_DATA_SOURCE )
        {
          errorMsg << "Too many variables found in the expression. "
                   << "A maximum of " << MAX_DATA_SOURCE << " are allowed."
                   << "Increase MAX_DATA_SOURCE to allow for more.";

          break;
        }
        else
        {
#if defined(USE_STD_CLASSES)
          mVariableList.push_back( variable );
          mNumVariables = mVariableList.size();
#else
          strcpy(mVariableList[mNumVariables++], variable);
#endif
        }
      }
    }
    // Invalid character.
    else
    {
      errorMsg << "Invalid expression: "
               << "Invalid character '" << c << "'.";
      break;
    }

    // Special case for a close parenthesis followed by an open
    // parenthesis with no operand between. Which implies a multiply.
    if( (mNodes[i_node    ].ID == NodeID::OPEN_PARENTHESIS ||
         mNodes[i_node    ].ID == NodeID::OPEN_ABS_BAR) &&
        (mNodes[i_previous].ID == NodeID::CLOSE_PARENTHESIS ||
         mNodes[i_previous].ID == NodeID::CLOSE_ABS_BAR) )
    {
      NodeInfo multInfo = NodeInfo::LeftAssociative;

      Plato::OrdinalType i_multNode = mNodesUsed++;
      mNodes[i_multNode].precedence  = 3;
      mNodes[i_multNode].ID          = NodeID::MULTIPLICATION;

      // Add the implict multiplication node to the tree.
      i_current = insertNode( i_current, i_multNode, multInfo );

      // Clear the close parenthesis node.
      clearNode( i_previous );

      // Prepare for the next iteration
      i_previous = i_multNode;

      // Now the open parenthesis node can be handled as usual.
    }

    // Some error handling for numbers and variables to make sure
    // there is an operand in between.
    if( (mNodes[i_node].ID    == NodeID::OPEN_PARENTHESIS   ||
         mNodes[i_node].ID    == NodeID::OPEN_ABS_BAR       ||
         mNodes[i_node].ID    == NodeID::NUMBER             ||
         mNodes[i_node].ID    == NodeID::VARIABLE)
        &&
        (mNodes[i_previous].ID == NodeID::CLOSE_PARENTHESIS ||
         mNodes[i_previous].ID == NodeID::CLOSE_ABS_BAR     ||
         mNodes[i_previous].ID == NodeID::FACTORIAL         ||
         mNodes[i_previous].ID == NodeID::NUMBER            ||
         mNodes[i_previous].ID == NodeID::VARIABLE) )
    {
      errorMsg << "Invalid expression: "
               << "Found a " << printNodeID( i_node, true )
               << " after a " << printNodeID( i_previous, true )
               << " without an operation in between.";
      break;
    }

    // Some error handling for positive and negative to make sure
    // there is a value in between.
    if( (mNodes[i_node].ID     == NodeID::POSITIVE ||
         mNodes[i_node].ID     == NodeID::NEGATIVE )
        &&
        (mNodes[i_previous].ID == NodeID::POSITIVE ||
         mNodes[i_previous].ID == NodeID::NEGATIVE) )
    {
      mNodes[i_node].ID
        = mNodes[i_node].ID     == NodeID::POSITIVE ? NodeID::ADDITION : NodeID::SUBTRACTION_;
      mNodes[i_previous].ID
        = mNodes[i_previous].ID == NodeID::POSITIVE ? NodeID::ADDITION : NodeID::SUBTRACTION_;

      errorMsg << "Invalid expression: Found a " << printNodeID( i_node, true )
                << " after a " << printNodeID( i_previous, true )
                << " without a value in between." << std::endl;
      break;
    }

    // Add the node to the tree.
    i_current = insertNode( i_current, i_node, info );

    if( i_current == (Plato::OrdinalType) -1 )
    {
      errorMsg << "Error: After inserting new node "
                << "the current node is now null.";
      break;
    }

    // Clear the close parenthesis and absolute value node. These are
    // not cleared until here because they are needed for the previous
    // checks.
    if( mNodes[i_previous].ID == NodeID::CLOSE_PARENTHESIS ||
        mNodes[i_previous].ID == NodeID::CLOSE_ABS_BAR)
      clearNode( i_previous );

    // Prepare for the next iteration
    i_previous = i_node;
  }

  // Clear the close parenthesis and absolute value node. These are
  // not cleared until here because they are needed for the previous
  // checks.
  if( mNodes[i_previous].ID == NodeID::CLOSE_PARENTHESIS ||
      mNodes[i_previous].ID == NodeID::CLOSE_ABS_BAR)
    clearNode( i_previous );

  // Found an error so add the original expression and a pointer to
  // the error.
  if( !errorMsg.str().empty() )
  {
    // Report the error.
    errorMsg << std::endl;

    // Print the original expression.
    errorMsg << expression << std::endl;

    // Print a pointer to where the error was found in the expression.
    int indent = strlen(expression) - strlen(expPtr) - 1;

    for( int i=0; i<indent; ++i )
      errorMsg << " ";

    errorMsg << "^";
  }
  else
  {
    // Make sure an open parenthesis was not left open.
    while( openParenLocations.size() )
    {
      if( errorMsg.str().size() )
        errorMsg << std::endl;

      // Report the error.
      errorMsg << "Invalid expression: Found an open '(' "
               << "' without a corresponding  close ')'."
               << std::endl;

      // Print the original expression.
      errorMsg << expression << std::endl;

      // Print a pointer to where the error was found in the expression.
      while( openParenLocations.back()-- )
        errorMsg << " ";

      errorMsg << "^" << std::endl;

      openParenLocations.pop_back();
    }

    // Make sure an open absolute value bar was not left open.
    while( openAbsBarLocations.size() )
    {
      if( errorMsg.str().size() )
        errorMsg << std::endl;

      // Report the error.
      errorMsg << "Invalid expression: Found an open '|' "
               << "without a corresponding close '|'."
               << std::endl;

      // Print the original expression.
      errorMsg << expression << std::endl;

      // Print a pointer to where the error was found in the expression.
      while( openAbsBarLocations.back()-- )
        errorMsg << " ";

      errorMsg << "^" << std::endl;

      openAbsBarLocations.pop_back();
    }
  }

  // Make sure two value functions are not empty.
  while( twoParamFunctionIDs.size() )
  {
    if( errorMsg.str().size() )
      errorMsg << std::endl;

    Plato::OrdinalType i_tmp = MAX_ARRAY_LENGTH - 1;
    mNodes[i_tmp].ID = twoParamFunctionIDs.back();

    // Report the error.
    errorMsg << "Invalid expression: "
             << "Found a two parameter function '"
             << printNodeID( i_tmp, true ) << "' with one or no parameters"
             << std::endl;

    // Print the original expression.
    errorMsg << expression << std::endl;

    // Print a pointer to where the error was found in the expression.
    int indent = twoParamFunctionLocations.back();

    for( int i=0; i<indent; ++i )
      errorMsg << " ";

    errorMsg << "^";

    twoParamFunctionIDs.pop_back();
    twoParamFunctionLocations.pop_back();
  }

  // Remove the initial open parenthesis '(' node as the root node.
  if( mNodes[i_root].i_right )
    mNodes[mNodes[i_root].i_right].i_parent = (Plato::OrdinalType) -1;
  else
  {
    errorMsg << "Invalid expression: "
             << "Empty expression.";
  }

  // Error print out the meesage and delete the tree.
  if( errorMsg.str().size() )
  {
    // Delete the tree.
    deleteNode( mNodes[i_root].i_right );
    mNodes[i_root].i_right = (Plato::OrdinalType) -1;

    ANALYZE_THROWERR( errorMsg.str() );
  }

  // Get the tru top level root node
  mTreeRootNode = mNodes[i_root].i_right;

  // Delete the temporary top level root.
  clearNode( i_root );

  // Take a subview so to redcue the memory footprint of the nodes.
  mNodes = subview(mNodes, std::make_pair(0, mNodesUsed-1));

  // Validate the resulting tree.
  if( valid_expression() == false )
  {
    std::stringstream errorMsg;
    errorMsg << "Could not validate the expression: " << expression;

    ANALYZE_THROWERR( errorMsg.str() );
  }

  Kokkos::Profiling::popRegion();

  // Traverse the resulting tree to obtain ancillary information for
  // post order evaluation and memory.
  traverse_expression();
 }

/******************************************************************************//**
 * \brief insertNode - Insert the node into the tree.
 * \param [in] i_current - index of the current node.
 * \param [in] i_new     - index of the new node.
 * \param [in] info      - associative information about the new node.
 * \return index - new current node.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
Plato::OrdinalType
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
insertNode(       Plato::OrdinalType i_current,
            const Plato::OrdinalType i_new,
            const NodeInfo info )
{
  // if( i_current == (Plato::OrdinalType) -1 )
  //   return (Plato::OrdinalType) -1;

  // if( mNodes[i_current].ID == NodeID::EMPTY_NODE )
  //   return (Plato::OrdinalType) -1;

  std::stringstream errorMsg;

  // Step 4: climb up to the parent node
  if( info == NodeInfo::NoInfo )
  {
    errorMsg << "Developer error: Can not add node, no node information.";

    i_current = (Plato::OrdinalType) -1;
  }
  else if( info == NodeInfo::RightAssociative )
  {
    // For right-associative
    while( i_current != (Plato::OrdinalType) -1 &&
           mNodes[i_current].precedence > mNodes[i_new].precedence )
      i_current = mNodes[i_current].i_parent;
  }
  else if( info == NodeInfo::LeftAssociative )
  {
    // For left-associative
    while( i_current != (Plato::OrdinalType) -1 &&
           mNodes[i_current].precedence >= mNodes[i_new].precedence )
      i_current = mNodes[i_current].i_parent;
  }
  else if( info == NodeInfo::SkipClimbUp )
  {
    // For open parenthesis, open absolute value, and positive/negative.
  }

  if( info != NodeInfo::NoInfo && i_current == (Plato::OrdinalType) -1 )
  {
    errorMsg << "Developer error: Can not add node, the current node is null.";
  }
  else if( mNodes[i_new].ID == NodeID::CLOSE_PARENTHESIS )
  {
    if( mNodes[i_current].i_parent != (Plato::OrdinalType) -1 )
    {
      // Step 5.1: get the parent of the '(' node.
      Plato::OrdinalType i_node = mNodes[i_current].i_parent;

      // Remove the current from between the parent and current's child
      mNodes[i_node].i_right = mNodes[i_current].i_right;

      // Now make the current's child point to the current's parent.
      if( mNodes[i_current].i_right != (Plato::OrdinalType) -1 )
        mNodes[mNodes[i_current].i_right].i_parent = i_node;

      // Step 5.2: delete the '(' node.
      clearNode( i_current );

      // Step 6: Set the 'current node' to be the parent node.
      i_current = i_node;
    }
    else
    {
      errorMsg << "Invalid expression: "
               << "Found a close parenthesis ')' without a "
               << "corresponding open parenthesis '('";

      i_current = (Plato::OrdinalType)-1;
    }
  }
  else if( mNodes[i_new].ID == NodeID::CLOSE_ABS_BAR )
  {
      // Change the open absolute value open bar '|' node to
      // an abs node.
      mNodes[i_current].ID = NodeID::ABS;

      // Step 5.1: get the parent of '(' node.
      Plato::OrdinalType i_node = mNodes[i_current].i_parent;

      // Step 6: Set the 'current node' to be the parent node.
      i_current = i_node;
  }
  else
  {
    // Step 5.1: create the new node
    Plato::OrdinalType i_node = i_new;

    mNodes[i_node].i_right = (Plato::OrdinalType) -1;

    // Step 5.2: add the new node
    mNodes[i_node].i_left = mNodes[i_current].i_right;

    if( mNodes[i_current].i_right != (Plato::OrdinalType) -1 )
      mNodes[mNodes[i_current].i_right].i_parent = i_node;

    mNodes[i_current].i_right = i_node;
    mNodes[i_node].i_parent = i_current;

    // Step 6: Set the 'current node' to be the new node
    i_current = i_node;
  }

  if( errorMsg.str().size() )
    ANALYZE_THROWERR( errorMsg.str() );

  return i_current;
}

/******************************************************************************//**
 * \brief commuteNode - Commute nodes so to make a left weighted tree.
 * \param [in] i_nore - index of the node.
 * \param [in] checkVariables - check for variable values.
 * \return NodeID - id of the bad node - EMPTY_NODE if okay.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
Plato::OrdinalType
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
commuteNode( const Plato::OrdinalType i_node )
{
  // This error should never happen, if it does it is a developer error.
  if( i_node == (Plato::OrdinalType) -1 )
  {
    std::stringstream errorMsg;
    errorMsg << "Invalid call to commuteNode - "
             << "node index is -1";
    ANALYZE_THROWERR( errorMsg.str() );
  }

  Node & node = mNodes[i_node];

  // Empty node. This should never happen as checks are made not to
  // evaluate empty nodes.
  if( node.ID == NodeID::EMPTY_NODE )
  {
    std::stringstream errorMsg;
    errorMsg << "Invalid call to commuteNode - "
             << "empty node index is " << i_node;
    ANALYZE_THROWERR( errorMsg.str() );
  }

  // Get the number of children on the left side of the tree.
  Plato::OrdinalType left = 0;

  switch( node.ID )
  {
    case NodeID::ADDITION:
    case NodeID::SUBTRACTION_:
    case NodeID::MULTIPLICATION:
    case NodeID::DIVISION:
    case NodeID::POWER:
//  case NodeID::FACTORIAL:
      left = commuteNode( node.i_left  );
      break;

    default:
      break;
  }

  // Get the number of children on the right side of the tree.
  Plato::OrdinalType right = 0;

  switch( node.ID )
  {
    case NodeID::POSITIVE:
    case NodeID::NEGATIVE:

    case NodeID::ADDITION:
    case NodeID::SUBTRACTION_:
    case NodeID::MULTIPLICATION:
    case NodeID::DIVISION:

    case NodeID::EXPONENTIAL:
    case NodeID::LOG:
    case NodeID::POWER:
    case NodeID::SQRT:
    case NodeID::ABS:

    case NodeID::SIN:
    case NodeID::COS:
    case NodeID::TAN:
      right = commuteNode( node.i_right );
      break;

    default:
      break;
  }

  // Check the current node for commutability.
  switch( node.ID )
  {
    // These nodes are commutable.
    case NodeID::ADDITION:
    case NodeID::MULTIPLICATION:

      // Swap the left and right children if the number of children on
      // the right is greater than the number of children on the left
      // as left wieghted trees utilize less memory when evaluated
      // without recursion.
      if( right > left )
      {
        Plato::OrdinalType tmp = node.i_right;
        node.i_right           = node.i_left;
        node.i_left            = tmp;
      }

    default:
      break;
  }

  // Return the number of number children plus one for this node.
  return std::max(left, right) + 1;
}

/******************************************************************************//**
 * \brief validateNode - Validate the current node and its children.
 * \param [in] i_nore - index of the node.
 * \param [in] checkVariables - check for variable values.
 * \return NodeID - id of the bad node - EMPTY_NODE if okay.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
typename ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::NodeID
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
validateNode( const Plato::OrdinalType i_node,
              const bool checkVariables ) const
{
  // This error should never happen, if it does it is a developer error.
  if( i_node == (Plato::OrdinalType) -1 )
  {
    std::stringstream errorMsg;
    errorMsg << "Invalid call to validateNode - "
             << "node index is -1";
    ANALYZE_THROWERR( errorMsg.str() );
  }

  const Node & node = mNodes[i_node];

  // Empty node. This should never happen as checks are made not to
  // evaluate empty nodes.
  if( node.ID == NodeID::EMPTY_NODE )
  {
    std::stringstream errorMsg;
    errorMsg << "Invalid call to validateNode - "
             << "empty node index is " << i_node;
    ANALYZE_THROWERR( errorMsg.str() );
  }

  // Validate the left side of the tree.
  switch( node.ID )
  {
    case NodeID::ADDITION:
    case NodeID::SUBTRACTION_:
    case NodeID::MULTIPLICATION:
    case NodeID::DIVISION:
    case NodeID::POWER:
//  case FACTORIAL:
    {
      NodeID left = validateNode( node.i_left  );
      if( left != NodeID::EMPTY_NODE )
      {
        return left;
      }
    }

      break;

    default:
      break;
  }

  // Validate the right side of the tree.
  switch( node.ID )
  {
    case NodeID::POSITIVE:
    case NodeID::NEGATIVE:

    case NodeID::ADDITION:
    case NodeID::SUBTRACTION_:
    case NodeID::MULTIPLICATION:
    case NodeID::DIVISION:

    case NodeID::EXPONENTIAL:
    case NodeID::LOG:
    case NodeID::POWER:
    case NodeID::SQRT:
    case NodeID::ABS:

    case NodeID::SIN:
    case NodeID::COS:
    case NodeID::TAN:
    {
      NodeID right = validateNode( node.i_right );
      if( right != NodeID::EMPTY_NODE )
      {
        return right;
      }
    }

      break;

    default:
      break;
  }

  Plato::OrdinalType thread = 0;

  std::stringstream errorMsg;
  NodeID nodeID = NodeID::EMPTY_NODE;

  // Validate the current node.
  switch( node.ID )
  {
    // These nodes need a valid left and right side value
    case NodeID::ADDITION:
    case NodeID::SUBTRACTION_:
    case NodeID::MULTIPLICATION:
    case NodeID::DIVISION:
    case NodeID::POWER:
      if( node.i_left == (Plato::OrdinalType) -1 )
      {
        errorMsg << "Invalid expression: No left side value for "
                  << printNodeID( i_node, true );
        nodeID = node.ID;
      }
      else if( node.i_right == (Plato::OrdinalType) -1 )
      {
        errorMsg << "Invalid expression: No right side value for "
                  << printNodeID( i_node, true );
        nodeID = node.ID;
      }
      else
        nodeID = NodeID::EMPTY_NODE;

      break;

    // These nodes need a valid right side value
    case NodeID::POSITIVE:
    case NodeID::NEGATIVE:

    case NodeID::EXPONENTIAL:
    case NodeID::LOG:
    case NodeID::SQRT:
    case NodeID::ABS:

    case NodeID::SIN:
    case NodeID::COS:
    case NodeID::TAN:
      if( node.i_right == (Plato::OrdinalType) -1 )
      {
        errorMsg << "Invalid expression: No value for "
                  << printNodeID( i_node, true );
        nodeID = node.ID;
      }
      else if( node.i_left != (Plato::OrdinalType) -1 )
      {
        errorMsg << "Invalid expression: Found a left side value for "
                  << printNodeID( i_node, true );
        nodeID = node.ID;
      }
      else
        nodeID = NodeID::EMPTY_NODE;

      break;

    // These nodes need a valid left side value
    case NodeID::FACTORIAL:
      if( node.i_left == (Plato::OrdinalType) -1 )
      {
        errorMsg << "Invalid expression: No value for "
                  << printNodeID( i_node, true );
        nodeID = node.ID;
      }
      else if( node.i_right != (Plato::OrdinalType) -1 )
      {
        errorMsg << "Invalid expression: Found a right side value for "
                  << printNodeID( i_node, true );
        nodeID = node.ID;
      }
      else
        nodeID = NodeID::EMPTY_NODE;

      break;

    case NodeID::VARIABLE:
      if( checkVariables )
      {
        Plato::OrdinalType type  = (Plato::OrdinalType) -1;
        Plato::OrdinalType index = (Plato::OrdinalType) -1;

        for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
        {
          if( STRCMP( mVariableMap(thread, i).key, node.variable ) == 0 )
          {
            if( mVariableMap(thread, i).value != (Plato::OrdinalType) -1 )
            {
              // Decode the value The type indicated the storage
              // container. There are three. The index gives the
              // location into the storage container being used.
              type  = mVariableMap(thread, i).value / MAX_DATA_SOURCE;
              index = mVariableMap(thread, i).value % MAX_DATA_SOURCE;
            }
            else
            {
              std::stringstream errorMsg;
              errorMsg << "Invalid call to validateNode - "
                       << "can not find values for variable: " << node.variable;
              ANALYZE_THROWERR( errorMsg.str() );
            }
          }
        }

        // Get the data from the storage container.
        if(  type == (Plato::OrdinalType) -1 ||
             type <  (Plato::OrdinalType)  0 ||
             type >  (Plato::OrdinalType)  2 ||
            index == (Plato::OrdinalType) -1 ||
            index <  (Plato::OrdinalType)  0 ||
            index >= (Plato::OrdinalType) mNumVariables )
        {
          std::stringstream errorMsg;
          errorMsg << "Invalid call to validateNode - "
                   << "can not find variable: " << node.variable << ". "
                   << "or bad type " << type << "  "
                   << "or bad index " << index << "  ";
          ANALYZE_THROWERR( errorMsg.str() );
        }

        nodeID = NodeID::EMPTY_NODE;
      }

      break;

    // None of these node should ever be in the tree
    case NodeID::OPEN_PARENTHESIS:
      errorMsg << "Invalid expression: Found an open parenthesis '(' without a "
                << "corresponding close parenthesis ')'";
      nodeID = node.ID;
      break;
    case NodeID::CLOSE_PARENTHESIS:
      errorMsg << "Invalid expression: Found a close parenthesis ')' without a "
                << "corresponding open parenthesis '('";
      nodeID = node.ID;
      break;
    case NodeID::OPEN_ABS_BAR:
      errorMsg << "Invalid expression: Found an open absolute value '|' without a "
                << "corresponding close absolute value '|'";
      nodeID = node.ID;
      break;
    case NodeID::CLOSE_ABS_BAR:
      errorMsg << "Invalid expression: Found a absolute value '|' without a "
                << "corresponding open absolute value '|'";
      nodeID = node.ID;
      break;

    default:
      nodeID = NodeID::EMPTY_NODE;
      break;
  }

  if( errorMsg.str().size() )
    ANALYZE_THROWERR( errorMsg.str() );

  return nodeID;
}

/******************************************************************************//**
 * \brief traverseNode - Post order traversal of the nodes.
 * \param [in] i_node - index of the node.
 * \param [in] depth  - depth of the node being evaluated.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
traverseNode( const Plato::OrdinalType i_node,
              const Plato::OrdinalType depth )
{
  // This error should never happen, if it does it is a developer error.
  if( i_node == (Plato::OrdinalType) -1 )
  {
    std::stringstream errorMsg;
    errorMsg << "Invalid call to traverseNode - "
             << "node index is -1";
    ANALYZE_THROWERR( errorMsg.str() );
  }

  Node & node = mNodes[i_node];

  // Empty node. This should never happen as checks are made not to
  // evaluate empty nodes.
  if( node.ID == NodeID::EMPTY_NODE )
  {
    std::stringstream errorMsg;
    errorMsg << "Invalid call to traverseNode - "
             << "empty node index is " << i_node;
    ANALYZE_THROWERR( errorMsg.str() );
  }

  // Add additional index(s) to the queue based on the depth of the node.
#if defined(USE_STD_CLASSES)
  while( mMemQueue.back() < depth )
    mMemQueue.push_back( mMemQueue.back()+1 );
#else
  while( mMemQueue[mMemQueueSize-1] < depth )
    mMemQueue[mMemQueueSize++] = mMemQueue[mMemQueueSize-1]+1;

  if(mMemQueueSize == MAX_NUM_NODES)
  {
    std::stringstream errorMsg;
    errorMsg << "The expression tree depth is too deep. "
             << "A maximum depth of " << MAX_NUM_NODES << " is allowed."
             << "Increase MAX_NUM_NODES to allow for a deeper expression tree.";
    ANALYZE_THROWERR( errorMsg.str() );
  }
#endif
  // Index to the chunk of temporary used by the left node.
  Plato::OrdinalType left_mem = -1;

  if( node.i_left != (Plato::OrdinalType) -1 )
  {
    traverseNode( node.i_left, depth+1 );
    left_mem = mNodes[node.i_left].i_memory;
  }

  // Index to the chunk of temporary used by the right node.
  Plato::OrdinalType right_mem = -1;

  if( node.i_right != (Plato::OrdinalType) -1 )
  {
    traverseNode( node.i_right, depth+1 );
    right_mem = mNodes[node.i_right].i_memory;
  }

  // Set the index of where the results will go using the first unused
  // index.
#if defined(USE_STD_CLASSES)
  node.i_memory = mMemQueue.front();
  mMemQueue.pop_front();
#else
  node.i_memory = mMemQueue[0];

  mMemQueueSize -= 1;
  for(Plato::OrdinalType i=0; i<mMemQueueSize; ++i)
    mMemQueue[i] = mMemQueue[i+1];
#endif

  // If data from the right node was used is can be now reused so push
  // the index to the front of the queue. Push right first as it was
  // last used.
  if( right_mem != (Plato::OrdinalType) -1 )
  {
#if defined(USE_STD_CLASSES)
    mMemQueue.push_front( right_mem );
#else
    mMemQueueSize += 1;
    for(Plato::OrdinalType i=1; i<mMemQueueSize; ++i)
      mMemQueue[i] = mMemQueue[i-1];

    mMemQueue[0] = right_mem;
#endif
  }

  // If data from the left node was used is can be now reused so push
  // the index to the front of the queue. Push left last as it was
  // first used.
  if( left_mem != (Plato::OrdinalType) -1 )
  {
#if defined(USE_STD_CLASSES)
    mMemQueue.push_front( left_mem );
#else
    mMemQueueSize += 1;
    for(Plato::OrdinalType i=1; i<mMemQueueSize; ++i)
      mMemQueue[i] = mMemQueue[i-1];

    mMemQueue[0] = left_mem;
#endif
  }

  // Add the node index to the post order evaluation order.
  mNodeOrder[mNodeCount++] = i_node;
}

/******************************************************************************//**
 * \brief factorial - Computes the factorial.
 * \param [in] n - number for factorial.
 * \return number - the factorial.
 **********************************************************************************/
/*
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
ResultType
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
factorial( ResultType n ) const
{
  // std::stringstream errorMsg;
  // errorMsg << "Invalid call to uninstantiated template factorial";
  // ANALYZE_THROWERR( errorMsg.str() );

  long ans = 1, m = (long) n;

  for( long i=1; i<=m; ++i )
    ans *= i;

  return (ResultType) ans;
}
*/

/******************************************************************************//**
 * \brief evaluateNode - Evaluate the current node.
 * \param [in] thread - thread being evaluated.
 * \param [in] i_node - index of the node.
 * \param [out] result - the expresion result.
 * \return evaluated - true if the node was evaluated.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
KOKKOS_INLINE_FUNCTION
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
evaluateNode( const Plato::OrdinalType thread,
              const Plato::OrdinalType i_node,
                    ResultType const & result ) const
{
  // printf( "%d %d %d %d \n", __LINE__, thread, mNumValues, i_node );

  // At this point no error checks are needed as the tree has been
  // validated as part of the parsing of the expression.

  // This error should never happen, if it does happen it is a
  // developer error. A similar check is made in traverseNode which
  // throws an error. Thus commented out.
  // if( i_node == (Plato::OrdinalType) -1 )
  // {
  //   GPU_WARNING("Invalid call to evaluateNode - ");
  //   prinf("node index is -1\n\n");

  //   return;
  // }
  const Node & node = mNodes[i_node];

  // Empty node. This should never happen as checks are made not to
  // evaluate empty nodes. A similar check is made in traverseNode
  // which throws an error. Thus commented out.
  // if( node.ID == NodeID::EMPTY_NODE )
  // {
  //   GPU_WARNING("Invalid call to evaluateNode - ");
  //   printf("node index is %d\n\n", i_node );

  //   return;
  // }

  // Get the left side of the tree.
  ResultType left;
  if( node.i_left != (Plato::OrdinalType) -1 )
  {
    left = mResults[ mNodes[node.i_left].i_memory];

#if USE_RECURSION
    // Recursion - not used because it creates lots of warnings and
    // recursions blows the stack on the GPU.
    evaluateNode( thread, node.i_left, left );
#endif
  }

  // Get the right side of the tree.
  ResultType right;
  if( node.i_right != (Plato::OrdinalType) -1 )
  {
    right = mResults[ mNodes[node.i_right].i_memory ];

#if USE_RECURSION
    // Recursion - not used because it creates lots of
    // warnings and recursions blows the stack on the GPU.
    evaluateNode( thread, node.i_right, right );
#endif
  }

  // Divide by zero check.
  // if( node.ID == DIVISION )
   // {
  //   for( Plato::OrdinalType i = 0; i < mNumValues; ++ i )
  //   {
  //     if( right(thread,i) == 0 )
  //     {
  //       PRINTERR("Warning: Divide by zero.");
  //       result(thread,i) = 0;
  //     }
  //   }
  // }

  // Do the operation
  switch( node.ID )
  {
    case NodeID::POSITIVE:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = +right(thread,i);
      break;
    case NodeID::NEGATIVE:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = -right(thread,i);
      break;

    case NodeID::ADDITION:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = left(thread,i) + right(thread,i);
      break;
    case NodeID::SUBTRACTION_:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = left(thread,i) - right(thread,i);
      break;
    case NodeID::MULTIPLICATION:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = left(thread,i) * right(thread,i);
      break;
    case NodeID::DIVISION:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        if( right(thread,i) == 0 )
          result(thread,i) = 0;
        else
          result(thread,i) = left(thread,i) / right(thread,i);
      break;

    case NodeID::EXPONENTIAL:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = exp(right(thread,i));
      break;
    case NodeID::LOG:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = log(right(thread,i));
      break;
    case NodeID::POWER:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = pow(left(thread,i), right(thread,i));
      break;
    case NodeID::SQRT:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = sqrt(right(thread,i));
      break;
    case NodeID::ABS:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = abs(right(thread,i));
      break;
    // case NodeID::FACTORIAL:
      // for( Plato::OrdinalType i=0; i<mNumValues; ++i )
      //        result(thread,i) = factorial(left(thread,i));
      // break;
    case NodeID::SIN:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = sin(right(thread,i));
      break;
    case NodeID::COS:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = cos(right(thread,i));
      break;
    case NodeID::TAN:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = tan(right(thread,i));
      break;

    case NodeID::NUMBER:
    {
      const Plato::Scalar value = node.number;

      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = value;

      break;
    }

    case NodeID::VARIABLE:
    {
      Plato::OrdinalType type  = (Plato::OrdinalType) -1;
      Plato::OrdinalType index = (Plato::OrdinalType) -1;

      for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
      {
        if( STRCMP( mVariableMap(thread,i).key, node.variable ) == 0 )
        {
          if( mVariableMap(thread,i).value != (Plato::OrdinalType) -1 )
          {
            // Decode the value The type indicated the storage
            // container. There are three. The index gives the location
            // into the storage container being used.
            type  = mVariableMap(thread,i).value / MAX_DATA_SOURCE;
            index = mVariableMap(thread,i).value % MAX_DATA_SOURCE;
          }
          else
          {
            // std::stringstream errorMsg;
            // errorMsg << "Invalid call to evaluateNode - "
            //       << "can not find values for variable: " << node.variable;
            // ANALYZE_THROWERR( errorMsg.str() );

            GPU_WARNING("Invalid call to evaluateNode - ");
            printf("can not find values for variable: %s.\n\n",
                   node.variable );
          }
        }
      }

      // Get the data from the storage container.
      if( type == SCALAR_DATA_SOURCE )
      {
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
        const ScalarType & value = mVariableScalarValues(thread, index);
#else
        const ScalarType & value = mVariableScalarValues(index);
#endif

        for( Plato::OrdinalType i=0; i<mNumValues; ++i )
          result(thread,i) = value;
      }
      else if( type == VECTOR_DATA_SOURCE )
      {
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
        const VectorType & values = mVariableVectorValues(thread, index);
#else
        const VectorType & values = mVariableVectorValues(index);
#endif
        // When the number of values is one index based on the thread
        if( mNumValues == 1 )
        {
          result(thread,0) = values[thread];
        }
        // Otherwise index over all values.
        else
        {
          for( Plato::OrdinalType i=0; i<mNumValues; ++i )
            result(thread,i) = values[i];
        }
      }
      else if( type == STATE_DATA_SOURCE )
      {
        const StateType & values = mVariableStateValues(index);

        for( Plato::OrdinalType i=0; i<mNumValues; ++i )
          result(thread,i) = values(thread,i);
      }
      // This error should never happen, if it does it is a developer error.
      else
      {
        // std::stringstream errorMsg;
        // errorMsg << "Invalid call to evaluateNode - "
        //       << "can not find storage container for variable: " << variable;
        // ANALYZE_THROWERR( errorMsg.str() );

        GPU_WARNING("Invalid call to evaluateNode - ");
        printf("can not find storage container for variable: %s.\n\n",
               node.variable );
      }

      break;
    }

    default:
      for( Plato::OrdinalType i=0; i<mNumValues; ++i )
        result(thread,i) = 0;
      break;
  }
}

/******************************************************************************//**
 * \brief deleteNode - Delete the current node.
 * \param [in] i_node - index of the node.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
deleteNode( const Plato::OrdinalType i_node )
{
  if( i_node == (Plato::OrdinalType) -1 )
    return;

  Node & node = mNodes[i_node];

  if( node.ID == NodeID::EMPTY_NODE )
    return;

  deleteNode( node.i_left  );
  deleteNode( node.i_right );

  clearNode( i_node );
}

/******************************************************************************//**
 * \brief clearNode - clear the current node.
 * \param [in] i_node - index of the node.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
clearNode( const Plato::OrdinalType i_node )
{
  if( i_node == (Plato::OrdinalType) -1 )
    return;

  Node & node = mNodes[i_node];

  node.ID          = NodeID::EMPTY_NODE;
  node.precedence  = 0;
  node.number      = 0;
  node.variable[0] = '\0';
  node.i_parent    = (Plato::OrdinalType) -1;
  node.i_right     = (Plato::OrdinalType) -1;
  node.i_left      = (Plato::OrdinalType) -1;
}

/******************************************************************************//**
 * \brief printNodeID - Print the current node's ID or its value/variable.
 * \param [in] i_node - index of the node.
 * \param [in] descriptor - in addition to the id print a descriptor
 * \return std::string - node id as a string
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
std::string
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
printNodeID( const Plato::OrdinalType i_node,
             const bool descriptor,
             const bool print_val ) const
{
  if( i_node == (Plato::OrdinalType) -1 )
    return "";

  const Node & node = mNodes[i_node];

  if( node.ID == NodeID::EMPTY_NODE )
    return "";

  Plato::OrdinalType thread = 0;

  std::stringstream os;

  os << "(" << i_node << ")  ";

  if( descriptor )
  {
    switch( node.ID )
    {
      case NodeID::NUMBER:             os << "number '";    break;
      case NodeID::VARIABLE:           os << "variable '";  break;
      case NodeID::OPEN_PARENTHESIS:   os << "open '";      break;
      case NodeID::CLOSE_PARENTHESIS:  os << "close '";     break;
      case NodeID::OPEN_ABS_BAR:       os << "open '";      break;
      case NodeID::CLOSE_ABS_BAR:      os << "close '";     break;
      default:
        os << "'";
    }
  }

  // Print the current node
  switch( node.ID )
  {
    case NodeID::POSITIVE:        os << "+ve";       break;
    case NodeID::NEGATIVE:        os << "-ve";       break;

    case NodeID::ADDITION:        os << "+";         break;
    case NodeID::SUBTRACTION_:    os << "-";         break;
    case NodeID::MULTIPLICATION:  os << "*";         break;
    case NodeID::DIVISION:        os << "/";         break;

    case NodeID::EXPONENTIAL:     os << "exp";       break;
    case NodeID::LOG:             os << "log";       break;
    case NodeID::POWER:           os << "pow";       break;
    case NodeID::SQRT:            os << "sqrt";      break;
    case NodeID::FACTORIAL:       os << "!";         break;
    case NodeID::ABS:             os << "abs";       break;

    case NodeID::SIN:             os << "sin";       break;
    case NodeID::COS:             os << "cos";       break;
    case NodeID::TAN:             os << "tan";       break;

    case NodeID::NUMBER:          os << node.number; break;
    case NodeID::VARIABLE:
    {
      os << node.variable;

      if( !print_val )
        break;

      Plato::OrdinalType type  = (Plato::OrdinalType) -1;
      Plato::OrdinalType index = (Plato::OrdinalType) -1;

      for( Plato::OrdinalType i=0; i<mNumVariables; ++i )
      {
        if( STRCMP( mVariableMap(thread, i).key, node.variable ) == 0 )
        {
          if( mVariableMap(thread, i).value != (Plato::OrdinalType) -1 )
          {
            // Decode the value The type indicated the storage
            // container. There are three. The index gives the location
            // into the storage container being used.
            type  = mVariableMap(thread, i).value / MAX_DATA_SOURCE;
            index = mVariableMap(thread, i).value % MAX_DATA_SOURCE;
          }
          else
          {
            std::stringstream errorMsg;
            errorMsg << "Invalid call to printNode - "
                     << "can not find values for variable: " << node.variable;
            ANALYZE_THROWERR( errorMsg.str() );
          }
        }
      }

      // Get the data from the storage container.
      if( type == SCALAR_DATA_SOURCE )
      {
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
        const ScalarType & value = mVariableScalarValues(thread, index);
#else
        const ScalarType & value = mVariableScalarValues(index);
#endif
        os << " = " << value << "  ";
      }
      else if( type == VECTOR_DATA_SOURCE )
      {
#if defined(USE_KOKKOS_VIEW_OF_VIEWS)
        const VectorType & values = mVariableVectorValues(thread, index);
#else
        const VectorType & values = mVariableVectorValues(index);
#endif
        os << " = " << values[0] << "  ";

        if( mNumThreads > 1 || mNumValues > 1 )
          os << "...  ";
      }
      else if( type == STATE_DATA_SOURCE )
      {
        const StateType & values = mVariableStateValues(index);

        os << " = " << values(0,0) << "  ";

        if( mNumThreads > 1 || mNumValues > 1 )
          os << "...  ";
      }
      else
      {
        std::stringstream errorMsg;
        errorMsg << "Invalid call to printNode - "
                 << "can not find variable: " << node.variable;
        ANALYZE_THROWERR( errorMsg.str() );
      }
    }
    break;

    case NodeID::OPEN_PARENTHESIS:   os << "(";      break;
    case NodeID::CLOSE_PARENTHESIS:  os << ")";      break;
    case NodeID::OPEN_ABS_BAR:       os << "|";      break;
    case NodeID::CLOSE_ABS_BAR:      os << "|";      break;

    default:
      os << "Error: Unknown id";
  }

  if( descriptor )
  {
    switch( node.ID )
    {
      case NodeID::NUMBER:          os << "'";  break;
      case NodeID::VARIABLE:        os << "'";  break;
      default:
        os << "'";
    }
  }

  return os.str();
}

/******************************************************************************//**
 * \brief printNode - Print the current node.
 * \param [in] os - the output stream
 * \param [in] i_node - index of the node.
 * \param [in] indent - number of spacs for indenting.
 * \param [in] print_val - print value(s) associated with variables.
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
void
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
printNode(       std::ostream &os,
           const Plato::OrdinalType i_node,
           const int indent,
           const bool print_val ) const
{
  if( i_node == (Plato::OrdinalType) -1 )
    return;

  const Node & node = mNodes[i_node];

  if( node.ID == NodeID::EMPTY_NODE )
    return;

  // Print right sub-tree
  printNode( os, node.i_right, indent+4, print_val );

  // Print the indent spaces
  for(int i=0; i<indent; ++i)
    os << " ";

  // Print the current node
  os << printNodeID( i_node, false, print_val );

  // Print the indent spaces
  // for(int i=0; i<2*indent; ++i)
  //   os << " ";

  os << std::endl;

  // Print left sub-tree
  printNode( os, node.i_left, indent+4, print_val );
}

/******************************************************************************//**
 * \brief printNode - Print the current node.
 * \param [in] i_node - index of the node.
 * \param [in] print_val - print value(s) associated with variables
 * \return std::string - node information as a string
 **********************************************************************************/
template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
std::string
ExpressionEvaluator<ResultType, StateType, VectorType, ScalarType>::
printNode( const Plato::OrdinalType i_node,
           const bool print_val ) const
{
  if( i_node == (Plato::OrdinalType) -1 )
    return "";

  const Node & node = mNodes[i_node];

  // if( node.ID == NodeID::EMPTY_NODE )
  //   return "";

  std::stringstream os;

  os << "Node[" << i_node << "] = ("
     << node.ID
     << "  '" << printNodeID(i_node, false, print_val) << "'  "
     << node.precedence     << "  "
     << node.number         << "  '"
     << node.variable       << "'  "
     << (int) node.i_parent << "  "
     << (int) node.i_right  << "  "
     << (int) node.i_left   << ")";

  return os.str();
}


template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
ResultType ExpressionEvaluator< ResultType, StateType,
                                VectorType, ScalarType >::sDummyResult;

template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
StateType ExpressionEvaluator< ResultType, StateType,
                               VectorType, ScalarType >::sDummyState;

template< typename ResultType, typename StateType,
          typename VectorType, typename ScalarType >
VectorType ExpressionEvaluator< ResultType, StateType,
                                VectorType, ScalarType >::sDummyVector;

} // namespace Plato

#endif
