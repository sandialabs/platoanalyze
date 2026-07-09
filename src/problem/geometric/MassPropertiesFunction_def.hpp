#pragma once

#include <memory>
#include <set>

#include "linear_algebra/PlatoEigen.hpp"
#include "parsing/ParseTools.hpp"
#include "problem/geometric/DivisionFunction.hpp"
#include "problem/geometric/GeometryScalarFunction.hpp"
#include "problem/geometric/LeastSquaresFunction.hpp"
#include "problem/geometric/MassMoment.hpp"
#include "problem/geometric/MassPropertiesFunction_decl.hpp"
#include "problem/geometric/WeightedSumFunction.hpp"
#include "utilities/AnalyzeMacros.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************/
/**
 * \brief Initialization of Mass Properties Function
 * \param [in] aMesh mesh database
 * \param [in] aProblemParams input parameters database
 **********************************************************************************/
template <typename PhysicsType>
void MassPropertiesFunction<PhysicsType>::initialize(Teuchos::ParameterList& aProblemParams)
{
    for (const auto& tDomain : mSpatialModel.mDomains)
    {
        auto tName = tDomain.domainName();

        auto tMaterialModels = aProblemParams.get<Teuchos::ParameterList>("Material Models");
        if (tMaterialModels.isSublist(tDomain.materialName()))
        {
            auto tMaterialModelInputs = tMaterialModels.sublist(tDomain.materialName());
            mMaterialDensities[tName] = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);
        }
    }
    createLeastSquaresFunction(mSpatialModel, aProblemParams);
}

/******************************************************************************/
/**
 * \brief Create the least squares mass properties function
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aProblemParams input parameters database
 **********************************************************************************/
template <typename PhysicsType>
void MassPropertiesFunction<PhysicsType>::createLeastSquaresFunction(const plato::domain::SpatialModel& aSpatialModel,
                                                                     Teuchos::ParameterList& aProblemParams)
{
    auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);

    auto tPropertyNamesArray = tFunctionParams.get<Teuchos::Array<std::string>>("Properties");
    auto tPropertyWeightsArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Weights");
    auto tPropertyGoldValuesArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>(
        "Gold Values", Teuchos::Array<Plato::Scalar>(tPropertyNamesArray.size(), 0.0));

    auto tPropertyNames = tPropertyNamesArray.toVector();
    auto tPropertyWeights = tPropertyWeightsArray.toVector();
    auto tPropertyGoldValues = tPropertyGoldValuesArray.toVector();

    mLeastSquaresExponent = Plato::ParseTools::getParam<unsigned int>(tFunctionParams, "Least Squares Exponent");

    if (tPropertyNames.size() != tPropertyWeights.size())
    {
        const std::string tErrorString = std::string("Number of 'Properties' in '") + mFunctionName +
                                         "' parameter list does not equal the number of 'Weights'";
        ANALYZE_THROWERR(tErrorString)
    }

    if (tPropertyNames.size() != tPropertyGoldValues.size())
    {
        const std::string tErrorString = std::string("Number of 'Gold Values' in '") + mFunctionName +
                                         "' parameter list does not equal the number of 'Properties'";
        ANALYZE_THROWERR(tErrorString)
    }

    if (mLeastSquaresExponent == 1 && std::any_of(tPropertyGoldValues.begin(), tPropertyGoldValues.end(),
                                                  [](double aEntry) { return aEntry != 0.0; }))
    {
        const std::string tErrorString = std::string("'Gold values' in '") + mFunctionName +
                                         "' parameter must be set to zero for a 'Least Squares Exponent' of 1";
        ANALYZE_THROWERR(tErrorString)
    }
    if (mLeastSquaresExponent != 1 && mLeastSquaresExponent != 2)
    {
        const std::string tErrorString =
            std::string("'Least Squares Exponent' in '") + mFunctionName +
            "' must be either 1 or 2. Found value: " + std::to_string(mLeastSquaresExponent);
        ANALYZE_THROWERR(tErrorString)
    }

    std::unordered_map<std::string, LeastSquaresFunctionData> tPropertyFunctions;
    constexpr Plato::Scalar tDefaultPropertyNormalization{1.0};
    for (Plato::OrdinalType tPropertyIndex = 0; tPropertyIndex < tPropertyNames.size(); ++tPropertyIndex)
    {
        tPropertyFunctions.try_emplace(tPropertyNames[tPropertyIndex], tPropertyWeights[tPropertyIndex],
                                       tPropertyGoldValues[tPropertyIndex], tDefaultPropertyNormalization, nullptr);
    }

    const bool tAllPropertiesSpecifiedByUser = allPropertiesSpecified(tPropertyNames);

    if (tAllPropertiesSpecifiedByUser)
        createAllMassPropertiesLeastSquaresFunction(aSpatialModel, std::move(tPropertyFunctions));
    else
        createItemizedLeastSquaresFunction(aSpatialModel, std::move(tPropertyFunctions));
}

/******************************************************************************/
/**
 * \brief Check if all properties were specified by user
 * \param [in] aPropertyNames names of properties specified by user
 * \return bool indicating if all properties were specified by user
 **********************************************************************************/
template <typename PhysicsType>
bool MassPropertiesFunction<PhysicsType>::allPropertiesSpecified(const std::vector<std::string>& aPropertyNames)
{
    // copy the vector since we sort it and remove items in this function
    std::vector<std::string> tPropertyNames(aPropertyNames.begin(), aPropertyNames.end());

    const unsigned int tUserSpecifiedNumberOfProperties = tPropertyNames.size();

    // Sort and erase duplicate entries
    std::sort(tPropertyNames.begin(), tPropertyNames.end());
    tPropertyNames.erase(std::unique(tPropertyNames.begin(), tPropertyNames.end()), tPropertyNames.end());

    // Check for duplicate entries from the user
    const unsigned int tUniqueNumberOfProperties = tPropertyNames.size();
    if (tUserSpecifiedNumberOfProperties != tUniqueNumberOfProperties)
    {
        ANALYZE_THROWERR("User specified mass properties vector contains duplicate entries!")
    }

    if (tUserSpecifiedNumberOfProperties < 10) return false;

    std::vector<std::string> tAllPropertiesVector = {"Mass", "CGx", "CGy", "CGz", "Ixx",
                                                     "Iyy",  "Izz", "Ixy", "Ixz", "Iyz"};
    std::sort(tAllPropertiesVector.begin(), tAllPropertiesVector.end());

    std::set<std::string> tAllPropertiesSet(tAllPropertiesVector.begin(), tAllPropertiesVector.end());
    std::set<std::string>::iterator tSetIterator;

    // if number of unqiue user-specified properties does not equal all of them, return false
    if (tPropertyNames.size() != tAllPropertiesVector.size()) return false;

    for (Plato::OrdinalType tIndex = 0; tIndex < tPropertyNames.size(); ++tIndex)
    {
        const std::string tCurrentProperty = tPropertyNames[tIndex];

        // Check to make sure it is a valid property
        tSetIterator = tAllPropertiesSet.find(tCurrentProperty);
        if (tSetIterator == tAllPropertiesSet.end())
        {
            const std::string tErrorString = std::string("Specified mass property '") + tCurrentProperty +
                                             "' not implemented. Options are: Mass, CGx, CGy, CGz, " +
                                             "Ixx, Iyy, Izz, Ixy, Ixz, Iyz";
            ANALYZE_THROWERR(tErrorString)
        }

        // property vectors were sorted so check that the properties match in sequence
        if (tCurrentProperty != tAllPropertiesVector[tIndex])
        {
            std::cout << std::format("Property {} does not equal property {} \n", tCurrentProperty.c_str(),
                                     tAllPropertiesVector[tIndex].c_str());
            std::cout << "If user specifies all mass properties, better performance may be experienced.\n";
            return false;
        }
    }

    return true;
}

template <typename PhysicsType>
auto MassPropertiesFunction<PhysicsType>::computePropertyNormalizationFromGoldValue(const Plato::Scalar aGoldValue)
    -> Plato::Scalar const
{
    const Plato::Scalar tAbsoluteGold = std::abs(aGoldValue);
    return tAbsoluteGold > mFunctionNormalizationCutoff ? tAbsoluteGold : 1.0;
}

template <typename PhysicsType>
auto MassPropertiesFunction<PhysicsType>::thresholdPropertyNormalization(const Plato::Scalar aNormalization)
    -> Plato::Scalar const
{
    const Plato::Scalar tAbsoluteNormalization = std::abs(aNormalization);
    return tAbsoluteNormalization > mFunctionNormalizationCutoff ? tAbsoluteNormalization
                                                                 : mFunctionNormalizationCutoff;
}

// CPD-OFF
template <typename PhysicsType>
void MassPropertiesFunction<PhysicsType>::createAllMassPropertiesLeastSquaresFunction(
    const plato::domain::SpatialModel& aSpatialModel,
    std::unordered_map<std::string, LeastSquaresFunctionData> aPropertyFunctions)
{
    std::cout << "Creating all mass properties function.\n";
    computeRotationAndParallelAxisTheoremMatrices(aPropertyFunctions);

    aPropertyFunctions.at("Mass").mScalarFunction = getMassFunction(aSpatialModel);
    aPropertyFunctions.at("Mass").mNormalization =
        computePropertyNormalizationFromGoldValue(aPropertyFunctions.at("Mass").mGoldValue);

    aPropertyFunctions.at("CGx").mScalarFunction = getFirstMomentOverMassRatio(aSpatialModel, "FirstX");
    aPropertyFunctions.at("CGx").mNormalization =
        computePropertyNormalizationFromGoldValue(aPropertyFunctions.at("CGx").mGoldValue);

    aPropertyFunctions.at("CGy").mScalarFunction = getFirstMomentOverMassRatio(aSpatialModel, "FirstY");
    aPropertyFunctions.at("CGy").mNormalization =
        computePropertyNormalizationFromGoldValue(aPropertyFunctions.at("CGy").mGoldValue);

    aPropertyFunctions.at("CGz").mScalarFunction = getFirstMomentOverMassRatio(aSpatialModel, "FirstZ");
    aPropertyFunctions.at("CGz").mNormalization =
        computePropertyNormalizationFromGoldValue(aPropertyFunctions.at("CGz").mGoldValue);

    aPropertyFunctions.at("Ixx").mScalarFunction = getMomentOfInertiaRotatedAboutCG(aSpatialModel, "XX");
    aPropertyFunctions.at("Ixx").mGoldValue = mInertiaPrincipalValues(0);
    aPropertyFunctions.at("Ixx").mNormalization =
        computePropertyNormalizationFromGoldValue(aPropertyFunctions.at("Ixx").mGoldValue);

    aPropertyFunctions.at("Iyy").mScalarFunction = getMomentOfInertiaRotatedAboutCG(aSpatialModel, "YY");
    aPropertyFunctions.at("Iyy").mGoldValue = mInertiaPrincipalValues(1);
    aPropertyFunctions.at("Iyy").mNormalization =
        computePropertyNormalizationFromGoldValue(aPropertyFunctions.at("Iyy").mGoldValue);

    aPropertyFunctions.at("Izz").mScalarFunction = getMomentOfInertiaRotatedAboutCG(aSpatialModel, "ZZ");
    aPropertyFunctions.at("Izz").mGoldValue = mInertiaPrincipalValues(2);
    aPropertyFunctions.at("Izz").mNormalization =
        computePropertyNormalizationFromGoldValue(aPropertyFunctions.at("Izz").mGoldValue);

    // Minimum Principal Moment of Inertia
    Plato::Scalar tMinPrincipalMoment =
        std::min(mInertiaPrincipalValues(0), std::min(mInertiaPrincipalValues(1), mInertiaPrincipalValues(2)));

    aPropertyFunctions.at("Ixy").mScalarFunction = getMomentOfInertiaRotatedAboutCG(aSpatialModel, "XY");
    aPropertyFunctions.at("Ixy").mGoldValue = 0.0;
    aPropertyFunctions.at("Ixy").mNormalization = thresholdPropertyNormalization(tMinPrincipalMoment);

    aPropertyFunctions.at("Ixz").mScalarFunction = getMomentOfInertiaRotatedAboutCG(aSpatialModel, "XZ");
    aPropertyFunctions.at("Ixz").mGoldValue = 0.0;
    aPropertyFunctions.at("Ixz").mNormalization = thresholdPropertyNormalization(tMinPrincipalMoment);

    aPropertyFunctions.at("Iyz").mScalarFunction = getMomentOfInertiaRotatedAboutCG(aSpatialModel, "YZ");
    aPropertyFunctions.at("Iyz").mGoldValue = 0.0;
    aPropertyFunctions.at("Iyz").mNormalization = thresholdPropertyNormalization(tMinPrincipalMoment);

    mLeastSquaresFunction = std::make_shared<Plato::Geometric::LeastSquaresFunction<PhysicsType>>(
        aSpatialModel, mDataMap, mLeastSquaresExponent);
    mLeastSquaresFunction->appendScalarFunctions(std::move(aPropertyFunctions));
}

template <typename PhysicsType>
void MassPropertiesFunction<PhysicsType>::computeRotationAndParallelAxisTheoremMatrices(
    const std::unordered_map<std::string, LeastSquaresFunctionData>& aPropertyFunctions)
{
    const Plato::Scalar Mass = aPropertyFunctions.at(std::string("Mass")).mGoldValue;

    const Plato::Scalar Ixx = aPropertyFunctions.at(std::string("Ixx")).mGoldValue;
    const Plato::Scalar Iyy = aPropertyFunctions.at(std::string("Iyy")).mGoldValue;
    const Plato::Scalar Izz = aPropertyFunctions.at(std::string("Izz")).mGoldValue;
    const Plato::Scalar Ixy = aPropertyFunctions.at(std::string("Ixy")).mGoldValue;
    const Plato::Scalar Ixz = aPropertyFunctions.at(std::string("Ixz")).mGoldValue;
    const Plato::Scalar Iyz = aPropertyFunctions.at(std::string("Iyz")).mGoldValue;

    const Plato::Scalar CGx = aPropertyFunctions.at(std::string("CGx")).mGoldValue;
    const Plato::Scalar CGy = aPropertyFunctions.at(std::string("CGy")).mGoldValue;
    const Plato::Scalar CGz = aPropertyFunctions.at(std::string("CGz")).mGoldValue;

    Plato::Array<3> tCGVector({CGx, CGy, CGz});

    const Plato::Scalar tNormSquared = Plato::dot(tCGVector, tCGVector);

    Plato::Matrix<3, 3> tParallelAxisTheoremMatrix =
        Plato::plus(Plato::identity<3>(tNormSquared), Plato::outer_product(tCGVector, tCGVector), -1.0);

    Plato::Matrix<3, 3> tGoldInertiaTensor({Ixx, Ixy, Ixz, Ixy, Iyy, Iyz, Ixz, Iyz, Izz});

    Plato::Matrix<3, 3> tGoldInertiaTensorAboutCG = Plato::plus(tGoldInertiaTensor, tParallelAxisTheoremMatrix, -Mass);

    Plato::decomposeEigenJacobi<3>(tGoldInertiaTensorAboutCG, mInertiaRotationMatrix, mInertiaPrincipalValues);

    std::cout << std::format("Eigenvalues of GoldInertiaTensor : {}, {}, {}\n", mInertiaPrincipalValues(0),
                             mInertiaPrincipalValues(1), mInertiaPrincipalValues(2));

    mMinusRotatedParallelAxisTheoremMatrix =
        Plato::times(-1.0, Plato::times(Plato::transpose(mInertiaRotationMatrix),
                                        Plato::times(tParallelAxisTheoremMatrix, mInertiaRotationMatrix)));
}
// CPD-ON

template <typename PhysicsType>
void MassPropertiesFunction<PhysicsType>::createItemizedLeastSquaresFunction(
    const plato::domain::SpatialModel& aSpatialModel,
    std::unordered_map<std::string, LeastSquaresFunctionData> aPropertyFunctions)
{
    std::cout << "Creating itemized mass properties function.\n";
    for (auto& [tPropertyName, tPropertyFunction] : aPropertyFunctions)
    {
        tPropertyFunction.mNormalization = computePropertyNormalizationFromGoldValue(tPropertyFunction.mGoldValue);
        if (tPropertyName == "Mass")
        {
            tPropertyFunction.mScalarFunction = getMassFunction(aSpatialModel);
        }
        else if (tPropertyName == "CGx")
        {
            tPropertyFunction.mScalarFunction = getFirstMomentOverMassRatio(aSpatialModel, "FirstX");
        }
        else if (tPropertyName == "CGy")
        {
            tPropertyFunction.mScalarFunction = getFirstMomentOverMassRatio(aSpatialModel, "FirstY");
        }
        else if (tPropertyName == "CGz")
        {
            tPropertyFunction.mScalarFunction = getFirstMomentOverMassRatio(aSpatialModel, "FirstZ");
        }
        else if (tPropertyName == "Ixx")
        {
            aPropertyFunctions.at("Ixx").mScalarFunction = getMomentOfInertia(aSpatialModel, "XX");
        }
        else if (tPropertyName == "Iyy")
        {
            aPropertyFunctions.at("Iyy").mScalarFunction = getMomentOfInertia(aSpatialModel, "YY");
        }
        else if (tPropertyName == "Izz")
        {
            aPropertyFunctions.at("Izz").mScalarFunction = getMomentOfInertia(aSpatialModel, "ZZ");
        }
        else if (tPropertyName == "Ixy")
        {
            aPropertyFunctions.at("Ixy").mScalarFunction = getMomentOfInertia(aSpatialModel, "XY");
        }
        else if (tPropertyName == "Ixz")
        {
            aPropertyFunctions.at("Ixz").mScalarFunction = getMomentOfInertia(aSpatialModel, "XZ");
        }
        else if (tPropertyName == "Iyz")
        {
            aPropertyFunctions.at("Iyz").mScalarFunction = getMomentOfInertia(aSpatialModel, "YZ");
        }
        else
        {
            const std::string tErrorString = std::string("Specified mass property '") + tPropertyName +
                                             "' not implemented. Options are: Mass, CGx, CGy, CGz, " +
                                             "Ixx, Iyy, Izz, Ixy, Ixz, Iyz";
            ANALYZE_THROWERR(tErrorString)
        }
    }

    mLeastSquaresFunction = std::make_shared<Plato::Geometric::LeastSquaresFunction<PhysicsType>>(
        aSpatialModel, mDataMap, mLeastSquaresExponent);
    mLeastSquaresFunction->appendScalarFunctions(std::move(aPropertyFunctions));
}

/******************************************************************************/
/**
 * \brief Create the mass function only
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \return physics scalar function
 **********************************************************************************/
template <typename PhysicsType>
std::shared_ptr<Plato::Geometric::GeometryScalarFunction<PhysicsType>>
MassPropertiesFunction<PhysicsType>::getMassFunction(const plato::domain::SpatialModel& aSpatialModel)
{
    std::shared_ptr<Plato::Geometric::GeometryScalarFunction<PhysicsType>> tMassFunction =
        std::make_shared<Plato::Geometric::GeometryScalarFunction<PhysicsType>>(aSpatialModel, mDataMap);
    tMassFunction->setFunctionName("Mass Function");

    std::string tCalculationType = std::string("Mass");

    for (const auto& tDomain : mSpatialModel.mDomains)
    {
        auto tName = tDomain.domainName();

        std::shared_ptr<Plato::Geometric::MassMoment<Residual>> tValue =
            std::make_shared<Plato::Geometric::MassMoment<Residual>>(tDomain, mDataMap);
        tValue->setMaterialDensity(mMaterialDensities[tName]);
        tValue->setCalculationType(tCalculationType);
        tMassFunction->setEvaluator(tValue, tName);

        std::shared_ptr<Plato::Geometric::MassMoment<GradientZ>> tGradientZ =
            std::make_shared<Plato::Geometric::MassMoment<GradientZ>>(tDomain, mDataMap);
        tGradientZ->setMaterialDensity(mMaterialDensities[tName]);
        tGradientZ->setCalculationType(tCalculationType);
        tMassFunction->setEvaluator(tGradientZ, tName);

        std::shared_ptr<Plato::Geometric::MassMoment<GradientX>> tGradientX =
            std::make_shared<Plato::Geometric::MassMoment<GradientX>>(tDomain, mDataMap);
        tGradientX->setMaterialDensity(mMaterialDensities[tName]);
        tGradientX->setCalculationType(tCalculationType);
        tMassFunction->setEvaluator(tGradientX, tName);
    }
    return tMassFunction;
}

/******************************************************************************/
/**
 * \brief Create the 'first mass moment divided by the mass' function (CG)
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aMomentType mass moment type (FirstX, FirstY, FirstZ)
 * \return scalar function base
 **********************************************************************************/
template <typename PhysicsType>
std::shared_ptr<Plato::Geometric::ScalarFunctionBase> MassPropertiesFunction<PhysicsType>::getFirstMomentOverMassRatio(
    const plato::domain::SpatialModel& aSpatialModel, const std::string& aMomentType)
{
    const std::string tNumeratorName = std::string("CG Numerator (Moment type = ") + aMomentType + ")";
    std::shared_ptr<Plato::Geometric::GeometryScalarFunction<PhysicsType>> tNumerator =
        std::make_shared<Plato::Geometric::GeometryScalarFunction<PhysicsType>>(aSpatialModel, mDataMap);
    tNumerator->setFunctionName(tNumeratorName);

    for (const auto& tDomain : mSpatialModel.mDomains)
    {
        auto tName = tDomain.domainName();

        std::shared_ptr<Plato::Geometric::MassMoment<Residual>> tNumeratorValue =
            std::make_shared<Plato::Geometric::MassMoment<Residual>>(tDomain, mDataMap);
        tNumeratorValue->setMaterialDensity(mMaterialDensities[tName]);
        tNumeratorValue->setCalculationType(aMomentType);
        tNumerator->setEvaluator(tNumeratorValue, tName);

        std::shared_ptr<Plato::Geometric::MassMoment<GradientZ>> tNumeratorGradientZ =
            std::make_shared<Plato::Geometric::MassMoment<GradientZ>>(tDomain, mDataMap);
        tNumeratorGradientZ->setMaterialDensity(mMaterialDensities[tName]);
        tNumeratorGradientZ->setCalculationType(aMomentType);
        tNumerator->setEvaluator(tNumeratorGradientZ, tName);

        std::shared_ptr<Plato::Geometric::MassMoment<GradientX>> tNumeratorGradientX =
            std::make_shared<Plato::Geometric::MassMoment<GradientX>>(tDomain, mDataMap);
        tNumeratorGradientX->setMaterialDensity(mMaterialDensities[tName]);
        tNumeratorGradientX->setCalculationType(aMomentType);
        tNumerator->setEvaluator(tNumeratorGradientX, tName);
    }

    const std::string tDenominatorName = std::string("CG Mass Denominator (Moment type = ") + aMomentType + ")";
    std::shared_ptr<Plato::Geometric::GeometryScalarFunction<PhysicsType>> tDenominator =
        getMassFunction(aSpatialModel);
    tDenominator->setFunctionName(tDenominatorName);

    std::shared_ptr<Plato::Geometric::DivisionFunction<PhysicsType>> tMomentOverMassRatioFunction =
        std::make_shared<Plato::Geometric::DivisionFunction<PhysicsType>>(aSpatialModel, mDataMap);
    tMomentOverMassRatioFunction->allocateNumeratorFunction(tNumerator);
    tMomentOverMassRatioFunction->allocateDenominatorFunction(tDenominator);
    tMomentOverMassRatioFunction->setFunctionName(std::string("CG ") + aMomentType);
    return tMomentOverMassRatioFunction;
}

/******************************************************************************/
/**
 * \brief Create the second mass moment function
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aMomentType second mass moment type (XX, XY, YY, ...)
 * \return scalar function base
 **********************************************************************************/
template <typename PhysicsType>
std::shared_ptr<Plato::Geometric::ScalarFunctionBase> MassPropertiesFunction<PhysicsType>::getSecondMassMoment(
    const plato::domain::SpatialModel& aSpatialModel, const std::string& aMomentType)
{
    const std::string tInertiaName = std::string("Second Mass Moment (Moment type = ") + aMomentType + ")";
    std::shared_ptr<Plato::Geometric::GeometryScalarFunction<PhysicsType>> tSecondMomentFunction =
        std::make_shared<Plato::Geometric::GeometryScalarFunction<PhysicsType>>(aSpatialModel, mDataMap);
    tSecondMomentFunction->setFunctionName(tInertiaName);

    for (const auto& tDomain : mSpatialModel.mDomains)
    {
        auto tName = tDomain.domainName();

        std::shared_ptr<Plato::Geometric::MassMoment<Residual>> tValue =
            std::make_shared<Plato::Geometric::MassMoment<Residual>>(tDomain, mDataMap);
        tValue->setMaterialDensity(mMaterialDensities[tName]);
        tValue->setCalculationType(aMomentType);
        tSecondMomentFunction->setEvaluator(tValue, tName);

        std::shared_ptr<Plato::Geometric::MassMoment<GradientZ>> tGradientZ =
            std::make_shared<Plato::Geometric::MassMoment<GradientZ>>(tDomain, mDataMap);
        tGradientZ->setMaterialDensity(mMaterialDensities[tName]);
        tGradientZ->setCalculationType(aMomentType);
        tSecondMomentFunction->setEvaluator(tGradientZ, tName);

        std::shared_ptr<Plato::Geometric::MassMoment<GradientX>> tGradientX =
            std::make_shared<Plato::Geometric::MassMoment<GradientX>>(tDomain, mDataMap);
        tGradientX->setMaterialDensity(mMaterialDensities[tName]);
        tGradientX->setCalculationType(aMomentType);
        tSecondMomentFunction->setEvaluator(tGradientX, tName);
    }

    return tSecondMomentFunction;
}

/******************************************************************************/
/**
 * \brief Create the moment of inertia function
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
 * \return scalar function base
 **********************************************************************************/
template <typename PhysicsType>
std::shared_ptr<Plato::Geometric::ScalarFunctionBase> MassPropertiesFunction<PhysicsType>::getMomentOfInertia(
    const plato::domain::SpatialModel& aSpatialModel, const std::string& aAxes)
{
    std::shared_ptr<Plato::Geometric::WeightedSumFunction<PhysicsType>> tMomentOfInertiaFunction =
        std::make_shared<Plato::Geometric::WeightedSumFunction<PhysicsType>>(aSpatialModel, mDataMap);
    tMomentOfInertiaFunction->setFunctionName(std::string("Inertia ") + aAxes);

    if (aAxes == "XX")
    {
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondYY"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondZZ"));
        tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        tMomentOfInertiaFunction->appendFunctionWeight(1.0);
    }
    else if (aAxes == "YY")
    {
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondXX"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondZZ"));
        tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        tMomentOfInertiaFunction->appendFunctionWeight(1.0);
    }
    else if (aAxes == "ZZ")
    {
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondXX"));
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondYY"));
        tMomentOfInertiaFunction->appendFunctionWeight(1.0);
        tMomentOfInertiaFunction->appendFunctionWeight(1.0);
    }
    else if (aAxes == "XY")
    {
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondXY"));
        tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
    }
    else if (aAxes == "XZ")
    {
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondXZ"));
        tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
    }
    else if (aAxes == "YZ")
    {
        tMomentOfInertiaFunction->allocateScalarFunctionBase(getSecondMassMoment(aSpatialModel, "SecondYZ"));
        tMomentOfInertiaFunction->appendFunctionWeight(-1.0);
    }
    else
    {
        const std::string tErrorString = std::string("Specified axes '") + aAxes +
                                         "' not implemented for moment of inertia calculation. " +
                                         "Options are: XX, YY, ZZ, XY, XZ, YZ";
        ANALYZE_THROWERR(tErrorString)
    }

    return tMomentOfInertiaFunction;
}

/******************************************************************************/
/**
 * \brief Create the moment of inertia function about the CG in the principal coordinate frame
 * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
 * \return scalar function base
 **********************************************************************************/
template <typename PhysicsType>
std::shared_ptr<Plato::Geometric::ScalarFunctionBase>
MassPropertiesFunction<PhysicsType>::getMomentOfInertiaRotatedAboutCG(const plato::domain::SpatialModel& aSpatialModel,
                                                                      const std::string& aAxes)
{
    std::shared_ptr<Plato::Geometric::WeightedSumFunction<PhysicsType>> tMomentOfInertiaFunction =
        std::make_shared<Plato::Geometric::WeightedSumFunction<PhysicsType>>(aSpatialModel, mDataMap);
    tMomentOfInertiaFunction->setFunctionName(std::string("InertiaRot ") + aAxes);

    std::vector<Plato::Scalar> tInertiaWeights(6);
    Plato::Scalar tMassWeight;

    getInertiaAndMassWeights(tInertiaWeights, tMassWeight, aAxes);
    for (unsigned int tIndex = 0; tIndex < 6; ++tIndex)
        tMomentOfInertiaFunction->appendFunctionWeight(tInertiaWeights[tIndex]);

    tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XX"));
    tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "YY"));
    tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "ZZ"));
    tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XY"));
    tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "XZ"));
    tMomentOfInertiaFunction->allocateScalarFunctionBase(getMomentOfInertia(aSpatialModel, "YZ"));

    tMomentOfInertiaFunction->allocateScalarFunctionBase(getMassFunction(aSpatialModel));
    tMomentOfInertiaFunction->appendFunctionWeight(tMassWeight);

    return tMomentOfInertiaFunction;
}

/******************************************************************************/
/**
 * \brief Compute the inertia weights and mass weight for the inertia about the CG rotated into principal frame
 * \param [out] aInertiaWeights inertia weights
 * \param [out] aMassWeight mass weight
 * \param [in] aAxes axes about which to compute the moment of inertia (XX, YY, ..)
 **********************************************************************************/
template <typename PhysicsType>
void MassPropertiesFunction<PhysicsType>::getInertiaAndMassWeights(std::vector<Plato::Scalar>& aInertiaWeights,
                                                                   Plato::Scalar& aMassWeight,
                                                                   const std::string& aAxes)
{
    const Plato::Scalar Q11 = mInertiaRotationMatrix(0, 0);
    const Plato::Scalar Q12 = mInertiaRotationMatrix(0, 1);
    const Plato::Scalar Q13 = mInertiaRotationMatrix(0, 2);

    const Plato::Scalar Q21 = mInertiaRotationMatrix(1, 0);
    const Plato::Scalar Q22 = mInertiaRotationMatrix(1, 1);
    const Plato::Scalar Q23 = mInertiaRotationMatrix(1, 2);

    const Plato::Scalar Q31 = mInertiaRotationMatrix(2, 0);
    const Plato::Scalar Q32 = mInertiaRotationMatrix(2, 1);
    const Plato::Scalar Q33 = mInertiaRotationMatrix(2, 2);

    if (aAxes == "XX")
    {
        aInertiaWeights[0] = Q11 * Q11;
        aInertiaWeights[1] = Q21 * Q21;
        aInertiaWeights[2] = Q31 * Q31;
        aInertiaWeights[3] = 2.0 * Q11 * Q21;
        aInertiaWeights[4] = 2.0 * Q11 * Q31;
        aInertiaWeights[5] = 2.0 * Q21 * Q31;

        aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0, 0);
    }
    else if (aAxes == "YY")
    {
        aInertiaWeights[0] = Q12 * Q12;
        aInertiaWeights[1] = Q22 * Q22;
        aInertiaWeights[2] = Q32 * Q32;
        aInertiaWeights[3] = 2.0 * Q12 * Q22;
        aInertiaWeights[4] = 2.0 * Q12 * Q32;
        aInertiaWeights[5] = 2.0 * Q22 * Q32;

        aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(1, 1);
    }
    else if (aAxes == "ZZ")
    {
        aInertiaWeights[0] = Q13 * Q13;
        aInertiaWeights[1] = Q23 * Q23;
        aInertiaWeights[2] = Q33 * Q33;
        aInertiaWeights[3] = 2.0 * Q13 * Q23;
        aInertiaWeights[4] = 2.0 * Q13 * Q33;
        aInertiaWeights[5] = 2.0 * Q23 * Q33;

        aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(2, 2);
    }
    else if (aAxes == "XY")
    {
        aInertiaWeights[0] = Q11 * Q12;
        aInertiaWeights[1] = Q21 * Q22;
        aInertiaWeights[2] = Q31 * Q32;
        aInertiaWeights[3] = Q11 * Q22 + Q12 * Q21;
        aInertiaWeights[4] = Q11 * Q32 + Q12 * Q31;
        aInertiaWeights[5] = Q21 * Q32 + Q22 * Q31;

        aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0, 1);
    }
    else if (aAxes == "XZ")
    {
        aInertiaWeights[0] = Q11 * Q13;
        aInertiaWeights[1] = Q21 * Q23;
        aInertiaWeights[2] = Q31 * Q33;
        aInertiaWeights[3] = Q11 * Q23 + Q13 * Q21;
        aInertiaWeights[4] = Q11 * Q33 + Q13 * Q31;
        aInertiaWeights[5] = Q21 * Q33 + Q23 * Q31;

        aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(0, 2);
    }
    else if (aAxes == "YZ")
    {
        aInertiaWeights[0] = Q12 * Q13;
        aInertiaWeights[1] = Q22 * Q23;
        aInertiaWeights[2] = Q32 * Q33;
        aInertiaWeights[3] = Q12 * Q23 + Q13 * Q22;
        aInertiaWeights[4] = Q12 * Q33 + Q13 * Q32;
        aInertiaWeights[5] = Q22 * Q33 + Q23 * Q32;

        aMassWeight = mMinusRotatedParallelAxisTheoremMatrix(1, 2);
    }
    else
    {
        const std::string tErrorString = std::string("Specified axes '") + aAxes +
                                         "' not implemented for inertia and mass weights calculation. " +
                                         "Options are: XX, YY, ZZ, XY, XZ, YZ";
        ANALYZE_THROWERR(tErrorString)
    }
}

/******************************************************************************/
/**
 * \brief Primary Mass Properties Function constructor
 * \param [in] aSpatialModel Plato Analyze spatial model
 * \param [in] aDataMap Plato Analyze data map
 * \param [in] aProblemParams input parameters database
 * \param [in] aName user defined function name
 **********************************************************************************/
template <typename PhysicsType>
MassPropertiesFunction<PhysicsType>::MassPropertiesFunction(const plato::domain::SpatialModel& aSpatialModel,
                                                            Plato::DataMap& aDataMap,
                                                            Teuchos::ParameterList& aProblemParams,
                                                            std::string& aName)
    : Plato::WorksetBase<typename PhysicsType::ElementType>(aSpatialModel.mMesh),
      mSpatialModel(aSpatialModel),
      mDataMap(aDataMap),
      mFunctionName(aName)
{
    initialize(aProblemParams);
}

/******************************************************************************/
/**
 * \brief Update physics-based parameters within optimization iterations
 * \param [in] aControl 1D view of control variables
 **********************************************************************************/
template <typename PhysicsType>
void MassPropertiesFunction<PhysicsType>::updateProblem(const Plato::ScalarVector& aControl) const
{
    mLeastSquaresFunction->updateProblem(aControl);
}

/******************************************************************************/
/**
 * \brief Evaluate Mass Properties Function
 * \param [in] aControl 1D view of control variables
 * \return scalar function evaluation
 **********************************************************************************/
template <typename PhysicsType>
Plato::Scalar MassPropertiesFunction<PhysicsType>::value(const Plato::ScalarVector& aControl) const
{
    Plato::Scalar tFunctionValue = mLeastSquaresFunction->value(aControl);
    return tFunctionValue;
}

/******************************************************************************/
/**
 * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the configuration parameters
 * \param [in] aControl 1D view of control variables
 * \return 1D view with the gradient of the scalar function wrt the configuration parameters
 **********************************************************************************/
template <typename PhysicsType>
Plato::ScalarVector MassPropertiesFunction<PhysicsType>::gradient_x(const Plato::ScalarVector& aControl) const
{
    Plato::ScalarVector tGradientX = mLeastSquaresFunction->gradient_x(aControl);
    return tGradientX;
}

/******************************************************************************/
/**
 * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the control variables
 * \param [in] aControl 1D view of control variables
 * \return 1D view with the gradient of the scalar function wrt the control variables
 **********************************************************************************/
template <typename PhysicsType>
Plato::ScalarVector MassPropertiesFunction<PhysicsType>::gradient_z(const Plato::ScalarVector& aControl) const
{
    Plato::ScalarVector tGradientZ = mLeastSquaresFunction->gradient_z(aControl);
    return tGradientZ;
}
}  // namespace Geometric

}  // namespace Plato
