#pragma once

#include "material/TensorConstant.hpp"
#include "material/Rank4VoigtConstant.hpp"
#include "material/Rank4SkewConstant.hpp"

#include "material/ScalarFunctor.hpp"
#include "material/TensorFunctor.hpp"
#include "material/Rank4VoigtFunctor.hpp"

#include "material/ScalarExpression.hpp"
#include "material/Rank4Field.hpp"
#include "material/Rank4FieldFactory.hpp"

#include "PlatoStaticsTypes.hpp"
#include "ParseTools.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato {

  enum class MaterialModelType { Linear, Nonlinear, Expression };

  /******************************************************************************/
  /*!
    \brief Base class for material models
  */
    template<int SpatialDim>
    class MaterialModel
  /******************************************************************************/
  {
      std::map<std::string, Plato::Scalar>                         mScalarConstantsMap;
      std::map<std::string, Plato::TensorConstant<SpatialDim>>     mTensorConstantsMap;
      std::map<std::string, Plato::Rank4VoigtConstant<SpatialDim>> mRank4VoigtConstantsMap;
      std::map<std::string, Plato::Rank4SkewConstant<SpatialDim>> mRank4SkewConstantsMap;

      std::map<std::string, Plato::ScalarFunctor>                 mScalarFunctorsMap;
      std::map<std::string, Plato::TensorFunctor<SpatialDim>>     mTensorFunctorsMap;
      std::map<std::string, Plato::Rank4VoigtFunctor<SpatialDim>> mRank4VoigtFunctorsMap;

      std::map<std::string, Plato::Rank4FieldFactory<SpatialDim>> mRank4FieldFactoryMap;

      Plato::MaterialModelType mType;
      std::string mExpression;

      bool mHasBasis;
      Plato::Matrix<SpatialDim, SpatialDim> mCartesianBasis;

    public:

      /******************************************************************************//**
       * \brief Default constructor for Plato::MaterialModel.
      **********************************************************************************/
      MaterialModel() : mType(Plato::MaterialModelType::Linear) {}

      /******************************************************************************//**
       * \brief Constructor for Plato::MaterialModel base class
       * \param [in] ParameterList with optional "Temperature Dependent" bool Parameter
       * unit test: PlatoMaterialModel_MaterialModel
      **********************************************************************************/
      MaterialModel(const Teuchos::ParameterList& aParamList) 
      {
          this->mType = Plato::MaterialModelType::Linear;
          if (aParamList.isType<bool>("Temperature Dependent"))
          {
              if (aParamList.get<bool>("Temperature Dependent")) {
                  this->mType = Plato::MaterialModelType::Nonlinear;
              }
          }
          if (aParamList.isSublist("Elastic Stiffness Expression")) 
          {
              this->mType = Plato::MaterialModelType::Expression;
              auto tCustomElasticSubList = aParamList.sublist("Elastic Stiffness Expression");
              if(tCustomElasticSubList.isType<double>("E0"))
              {          
                  this->setScalarConstant("E0", tCustomElasticSubList.get<double>("E0"));
              }
              if(tCustomElasticSubList.isType<std::string>("Expression"))
              {          
                  this->expression(tCustomElasticSubList.get<std::string>("Expression"));
              }
              if(tCustomElasticSubList.isType<double>("Poissons Ratio"))
              {          
                  this->setScalarConstant("Poissons Ratio", tCustomElasticSubList.get<double>("Poissons Ratio"));
              }
              if(tCustomElasticSubList.isType<double>("Density"))
              {          
                  this->setScalarConstant("Density", tCustomElasticSubList.get<double>("Density"));
              }
          }

          parseCartesianBasis(aParamList);
      }

      void parseCartesianBasis(const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isSublist("Basis")) 
          {
              Plato::ParseTools::getBasis(aParamList, mCartesianBasis);
              mHasBasis = true;
          }
          else
          {
              mHasBasis = false;
          }
      }

      Plato::MaterialModelType type() const { return this->mType; }
      std::string expression() const { return this->mExpression; }
      void expression(const std::string aString) { this->mExpression = aString; }

      // getters
      //

      Plato::Matrix<SpatialDim, SpatialDim>
      getCartesianBasis() const { return mCartesianBasis; }

      bool hasCartesianBasis() const { return mHasBasis; }

      // scalar constant
      bool scalarConstantExists(std::string aConstantName)
      { return mScalarConstantsMap.count(aConstantName) == 1 ? true : false; }

      Plato::Scalar getScalarConstant(std::string aConstantName)
      { return mScalarConstantsMap[aConstantName]; }

      // Tensor constant
      Plato::TensorConstant<SpatialDim> getTensorConstant(std::string aConstantName)
      { return mTensorConstantsMap[aConstantName]; }

      // Rank4Voigt constant
      Plato::Rank4VoigtConstant<SpatialDim> getRank4VoigtConstant(std::string aConstantName)
      { 
        if (mRank4VoigtConstantsMap.count(aConstantName) == 0)
          ANALYZE_THROWERR("Constant with name: " + aConstantName + " not found in Rank4VoigtConstantsMap");
        return mRank4VoigtConstantsMap[aConstantName]; 
      }

      Plato::Rank4SkewConstant<SpatialDim> getRank4SkewConstant(std::string aConstantName)
      {
        if (mRank4SkewConstantsMap.count(aConstantName) == 0)
          ANALYZE_THROWERR("Constant with name: " + aConstantName + " not found in mRank4SkewConstantsMap");
        return mRank4SkewConstantsMap[aConstantName];
      }

      // scalar functor
      Plato::ScalarFunctor getScalarFunctor(std::string aFunctorName)
      { return mScalarFunctorsMap[aFunctorName]; }

      // tensor functor
      Plato::TensorFunctor<SpatialDim> getTensorFunctor(std::string aFunctorName)
      { return mTensorFunctorsMap[aFunctorName]; }

      // Rank4Voigt functor
      Plato::Rank4VoigtFunctor<SpatialDim> getRank4VoigtFunctor(std::string aFunctorName)
      { return mRank4VoigtFunctorsMap[aFunctorName]; }

      // Rank4Voigt field
      template<typename EvaluationType>
      std::shared_ptr<Plato::Rank4Field<EvaluationType>> getRank4Field(std::string aFieldName)
      {
          if(mRank4FieldFactoryMap.count(aFieldName) == 0)
          {
              std::stringstream err;
              err << "Attempted to retrieve non-extistant Rank4FieldFactory with name " << aFieldName;
              ANALYZE_THROWERR(err.str());
          }
          auto tFactory = mRank4FieldFactoryMap.at(aFieldName);
          return tFactory.template create<EvaluationType>();
      }


      // setters
      //

      // scalar constant
      void setScalarConstant(std::string aConstantName, Plato::Scalar aConstantValue)
      { mScalarConstantsMap[aConstantName] = aConstantValue; }

      // tensor constant
      void setTensorConstant(std::string aConstantName, Plato::TensorConstant<SpatialDim> aConstantValue)
      { mTensorConstantsMap[aConstantName] = aConstantValue; }

      // Rank4Voigt constant
      void setRank4VoigtConstant(std::string aConstantName, Plato::Rank4VoigtConstant<SpatialDim> aConstantValue)
      { mRank4VoigtConstantsMap[aConstantName] = aConstantValue; }

      void setRank4SkewConstant(std::string aConstantName, Plato::Rank4SkewConstant<SpatialDim> aConstantValue)
      { mRank4SkewConstantsMap[aConstantName] = aConstantValue; }

      // scalar functor
      void setScalarFunctor(std::string aFunctorName, Plato::ScalarFunctor aFunctorValue)
      { mScalarFunctorsMap[aFunctorName] = aFunctorValue; }

      // tensor functor
      void setTensorFunctor(std::string aFunctorName, Plato::TensorFunctor<SpatialDim> aFunctorValue)
      { mTensorFunctorsMap[aFunctorName] = aFunctorValue; }

      // Rank4Voigt functor
      void setRank4VoigtFunctor(std::string aFunctorName, Plato::Rank4VoigtFunctor<SpatialDim> aFunctorValue)
      { mRank4VoigtFunctorsMap[aFunctorName] = aFunctorValue; }

      // Rank4Voigt field
      void setRank4Field(std::string aFieldName, Plato::Rank4FieldFactory<SpatialDim> aFieldValue)
      { mRank4FieldFactoryMap[aFieldName] = aFieldValue; }


      /******************************************************************************/
      /*!
        \brief create either scalar constant or scalar functor from input
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void parseScalar(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
            auto tValue= aParamList.get<Plato::Scalar>(aName);
            if (this->mType == Plato::MaterialModelType::Nonlinear)
            {
                this->setScalarFunctor(aName, Plato::ScalarFunctor(tValue));
            }
            else
            {
                this->setScalarConstant(aName, tValue);
            }
          }
          else
          if( aParamList.isSublist(aName) )
          {
            if (this->mType == Plato::MaterialModelType::Linear)
            {
                std::stringstream err;
                err << "Found a temperature dependent constant in a linear model." << std::endl;
                err << "Models must be declared temperature dependent." << std::endl;
                err << "Set Parameter 'temperature dependent' to 'true'." << std::endl;
                ANALYZE_THROWERR(err.str());
            }
            auto tList = aParamList.sublist(aName);
            this->setScalarFunctor(aName, Plato::ScalarFunctor(tList));
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as either a Parameter or ParameterList";
              ANALYZE_THROWERR(err.str());
          }
      }
      /******************************************************************************/
      /*!
        \brief create scalar constant.  Add default if not found
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void
      parseScalarConstant(
        std::string aName,
        const Teuchos::ParameterList& aParamList,
        Plato::Scalar aDefaultValue)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
              auto tValue= aParamList.get<Plato::Scalar>(aName);
              this->setScalarConstant(aName, tValue);
          }
          else
          {
              this->setScalarConstant(aName, aDefaultValue);
          }
      }

      /******************************************************************************/
      /*!
        \brief create scalar constant.  Throw if not found.
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void parseScalarConstant(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
            auto tValue= aParamList.get<Plato::Scalar>(aName);
            this->setScalarConstant(aName, tValue);
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as a Parameter of type 'double'";
              ANALYZE_THROWERR(err.str());
          }
      }

      /******************************************************************************/
      /*!
        \brief create either tensor constant or tensor functor from input
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void parseTensor(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
            auto tValue= aParamList.get<Plato::Scalar>(aName);
            if (this->mType == Plato::MaterialModelType::Nonlinear)
            {
                this->setTensorFunctor(aName, Plato::TensorFunctor<SpatialDim>(tValue));
            }
            else
            {
                this->setTensorConstant(aName, Plato::TensorConstant<SpatialDim>(tValue));
            }
          }
          else
          if( aParamList.isSublist(aName) )
          {
            auto tList = aParamList.sublist(aName);
            if (this->mType == Plato::MaterialModelType::Nonlinear)
            {
                this->setTensorFunctor(aName, Plato::TensorFunctor<SpatialDim>(tList));
            }
            else
            {
                this->setTensorConstant(aName, Plato::TensorConstant<SpatialDim>(tList));
            }
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as either a Parameter or ParameterList";
              ANALYZE_THROWERR(err.str());
          }
      }

      /******************************************************************************/
      /*!
        \brief create either Rank4Voigt constant or Rank4Voigt functor from input
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void parseRank4Voigt(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if( aParamList.isSublist(aName) )
          {
              auto tList = aParamList.sublist(aName);
              if (this->mType == Plato::MaterialModelType::Linear)
              {
                  this->setRank4VoigtConstant(aName, Plato::Rank4VoigtConstant<SpatialDim>(tList));
              }
              else
              {
                  this->setRank4VoigtFunctor(aName, Plato::Rank4VoigtFunctor<SpatialDim>(tList));
              }
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as a ParameterList";
              ANALYZE_THROWERR(err.str());
          }
      }

      /******************************************************************************/
      /*!
        \brief create either Rank4Voigt constant or Rank4Voigt functor from input
        unit test: PlatoMaterialModel_MaterialModel
      */
      /******************************************************************************/
      void 
      parseRank4Field
      (const std::string& aListName, 
       const Teuchos::ParameterList& aParamList,
       const std::string& aName = "")
      {
          const std::string& tName = aName.empty() ? aListName : aName;
          if( aParamList.isSublist(aListName) )
          {
              auto tList = aParamList.sublist(aListName);
              this->setRank4Field(tName, Plato::Rank4FieldFactory<SpatialDim>(tList));
          }
          else
          {
              std::stringstream err;
              err << "Required input missing. '" << aName
                  << "' must be provided as a ParameterList";
              ANALYZE_THROWERR(err.str());
          }
      }

      // Others
      bool hasRank4VoigtConstant(const std::string& aConstantName)
      { 
        if (mRank4VoigtConstantsMap.count(aConstantName) == 0)
          return false;
        return true; 
      }

      bool hasRank4Field(const std::string& aFieldName)
      { 
        if (mRank4FieldFactoryMap.count(aFieldName) == 0)
          return false;
        return true; 
      }

      void
      setModelType()
      {
          if(mRank4VoigtFunctorsMap.size() > 0)
          {
              mType = Plato::MaterialModelType::Nonlinear;
          }
          else if(mRank4FieldFactoryMap.size() > 0)
          {
              mType = Plato::MaterialModelType::Expression;
              if(mRank4VoigtConstantsMap.size() > 0)
              {
                  ANALYZE_THROWERR("MaterialModel has Expression rank4 tensors and Linear rank4 tensors. Only one kind should be used.");
              }
              if(mRank4VoigtFunctorsMap.size() > 0)
              {
                  ANALYZE_THROWERR("MaterialModel has Expression rank4 tensors and Nonlinear rank4 tensors. Only one kind should be used.");
              }
          }
      }
  };
} // namespace Plato
