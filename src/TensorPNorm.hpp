#ifndef TENSOR_P_NORM_HPP
#define TENSOR_P_NORM_HPP

#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Tensor p-norm functor.

 Given a voigt tensors, compute the p-norm.
 Assumes single point integration.
 */
/******************************************************************************/
template<Plato::OrdinalType VoigtLength>
class TensorPNormFunctor
{
public:

    TensorPNormFunctor(Teuchos::ParameterList& params)
    {
    }

    template<typename ResultScalarType, typename TensorScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(Plato::OrdinalType cellOrdinal,
                                       Plato::ScalarVectorT<ResultScalarType> pnorm,
                                       Plato::ScalarMultiVectorT<TensorScalarType> voigtTensor,
                                       Plato::OrdinalType p,
                                       Plato::ScalarVectorT<VolumeScalarType> cellVolume) const
    {

        // compute scalar product
        //
        pnorm(cellOrdinal) = 0.0;
        for(Plato::OrdinalType iVoigt = 0; iVoigt < VoigtLength; iVoigt++)
        {
            pnorm(cellOrdinal) += voigtTensor(cellOrdinal, iVoigt) * voigtTensor(cellOrdinal, iVoigt);
        }
        pnorm(cellOrdinal) = pow(pnorm(cellOrdinal), p / 2.0);
        pnorm(cellOrdinal) *= cellVolume(cellOrdinal);
    }
};
// class TensorPNormFunctor

/******************************************************************************/
/*! Weighted tensor p-norm functor.

 Given a voigt tensors, compute the weighted p-norm.
 Assumes single point integration.
 */
/******************************************************************************/
template<Plato::OrdinalType VoigtLength>
class WeightedNormFunctor
{

    Plato::Scalar mScalarProductWeight;
    Plato::Scalar mReferenceWeight;

public:

    WeightedNormFunctor(Teuchos::ParameterList& params) :
            mScalarProductWeight(1.0),
            mReferenceWeight(1.0)
    {
        auto p = params.sublist("Normalize").sublist("Scalar");
        if(p.isType<Plato::Scalar>("Reference Weight"))
        {
            mReferenceWeight = p.get<Plato::Scalar>("Reference Weight");
        }
        if(p.isType<Plato::Scalar>("Scalar Product Weight"))
        {
            mScalarProductWeight = p.get<Plato::Scalar>("Scalar Product Weight");
        }
    }

    template<typename ResultScalarType, typename TensorScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(Plato::OrdinalType cellOrdinal,
                                       Plato::ScalarVectorT<ResultScalarType> pnorm,
                                       Plato::ScalarMultiVectorT<TensorScalarType> voigtTensor,
                                       Plato::OrdinalType p,
                                       Plato::ScalarVectorT<VolumeScalarType> cellVolume) const
    {

        // compute scalar product
        //
        pnorm(cellOrdinal) = 0.0;
        for(Plato::OrdinalType iVoigt = 0; iVoigt < VoigtLength; iVoigt++)
        {
            pnorm(cellOrdinal) += voigtTensor(cellOrdinal, iVoigt) * voigtTensor(cellOrdinal, iVoigt);
        }
        pnorm(cellOrdinal) = sqrt(mScalarProductWeight * pnorm(cellOrdinal));
        pnorm(cellOrdinal) /= mReferenceWeight;

        pnorm(cellOrdinal) = pow(pnorm(cellOrdinal), p);
        pnorm(cellOrdinal) *= cellVolume(cellOrdinal);
    }
};
// class WeightedNormFunctor

/******************************************************************************/
/*! Barlat tensor p-norm functor.

 Given a voigt tensors, compute the Barlat p-norm.
 Assumes single point integration.
 */
/******************************************************************************/
template<Plato::OrdinalType VoigtLength>
class BarlatNormFunctor
{
    Plato::OrdinalType mBarlatExponent;
    Plato::Scalar mReferenceWeight;

    static constexpr Plato::OrdinalType barlatLength = 6;
    Plato::Scalar mCp[barlatLength][barlatLength];
    Plato::Scalar mCpp[barlatLength][barlatLength];

public:

    BarlatNormFunctor(Teuchos::ParameterList& params) :
            mBarlatExponent(6),
            mReferenceWeight(1.0)
    {

        auto p = params.sublist("Normalize").sublist("Barlat");
        if(p.isType < Plato::OrdinalType > ("Barlat Exponent"))
        {
            mBarlatExponent = p.get<Plato::Scalar>("Barlat Exponent");
        }
        if(p.isType<Plato::Scalar>("Reference Weight"))
        {
            mReferenceWeight = p.get<Plato::Scalar>("Reference Weight");
        }

        Plato::Scalar cp[barlatLength][barlatLength];
        Plato::Scalar cpp[barlatLength][barlatLength];

        for(Plato::OrdinalType i = 0; i < barlatLength; i++)
        {
            std::stringstream ss;
            ss << "T1" << i;
            auto vals = p.get<Teuchos::Array<Plato::Scalar>>(ss.str());
            for(Plato::OrdinalType j = 0; j < barlatLength; j++)
                cp[i][j] = vals[j];
        }

        for(Plato::OrdinalType i = 0; i < barlatLength; i++)
        {
            std::stringstream ss;
            ss << "T2" << i;
            auto vals = p.get<Teuchos::Array<Plato::Scalar>>(ss.str());
            for(Plato::OrdinalType j = 0; j < barlatLength; j++)
                cpp[i][j] = vals[j];
        }

        Plato::Scalar mapToDevStress[barlatLength][barlatLength] = {2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0,
                                                                    -1.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0,
                                                                    -1.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0, 0.0,
                                                                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};

        for(Plato::OrdinalType i = 0; i < barlatLength; i++)
        {
            for(Plato::OrdinalType j = 0; j < barlatLength; j++)
            {
                mCp[i][j] = 0.0;
                mCpp[i][j] = 0.0;
                for(Plato::OrdinalType k = 0; k < barlatLength; k++)
                {
                    mCp[i][j] += cp[i][k] * mapToDevStress[k][j];
                    mCpp[i][j] += cpp[i][k] * mapToDevStress[k][j];
                }
            }
        }

    }

    template<typename ResultScalarType, typename TensorScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(Plato::OrdinalType cellOrdinal,
                                       Plato::ScalarVectorT<ResultScalarType> pnorm,
                                       Plato::ScalarMultiVectorT<TensorScalarType> voigtTensor,
                                       Plato::OrdinalType p,
                                       Plato::ScalarVectorT<VolumeScalarType> cellVolume) const
    {
        TensorScalarType s[barlatLength] = {0.0};
        if(VoigtLength == 6)
        {
            for(Plato::OrdinalType i = 0; i < VoigtLength; i++)
                s[i] = voigtTensor(cellOrdinal, i);
        }
        else if(VoigtLength == 3)
        {
            s[0] = voigtTensor(cellOrdinal, 0);
            s[1] = voigtTensor(cellOrdinal, 1);
            s[5] = voigtTensor(cellOrdinal, 2);
        }
        else if(VoigtLength == 1)
        {
            s[0] = voigtTensor(cellOrdinal, 0);
        }

        TensorScalarType sp[barlatLength], spp[barlatLength];
        for(Plato::OrdinalType i = 0; i < barlatLength; i++)
        {
            sp[i] = 0.0;
            spp[i] = 0.0;
            for(Plato::OrdinalType j = 0; j < barlatLength; j++)
            {
                sp[i] += mCp[i][j] * s[j];
                spp[i] += mCpp[i][j] * s[j];
            }
        }

        //Get invariates of the deviators. appendix A of Barlat, 2004.
        TensorScalarType Hp1 = (sp[0] + sp[1] + sp[2]) / 3.0;
        TensorScalarType Hp2 = (sp[3] * sp[3] + sp[4] * sp[4] + sp[5] * sp[5] - sp[1] * sp[2] - sp[2] * sp[0]
                                - sp[0] * sp[1])
                               / 3.0;
        TensorScalarType Hp3 = (2.0 * sp[3] * sp[4] * sp[5] + sp[0] * sp[1] * sp[2] - sp[0] * sp[3] * sp[3]
                                - sp[1] * sp[4] * sp[4] - sp[2] * sp[5] * sp[5])
                               / 2.0;
        TensorScalarType Hpp1 = (spp[0] + spp[1] + spp[2]) / 3.0;
        TensorScalarType Hpp2 = (spp[3] * spp[3] + spp[4] * spp[4] + spp[5] * spp[5] - spp[1] * spp[2] - spp[2] * spp[0]
                                 - spp[0] * spp[1])
                                / 3.0;
        TensorScalarType Hpp3 = (2.0 * spp[3] * spp[4] * spp[5] + spp[0] * spp[1] * spp[2] - spp[0] * spp[3] * spp[3]
                                 - spp[1] * spp[4] * spp[4] - spp[2] * spp[5] * spp[5])
                                / 2.0;

        //Get interim values for p, q, and theta. These are defined analytically in the Barlat text.
        TensorScalarType Pp = 0.0;
        if(Hp1 * Hp1 + Hp2 > 0.0)
            Pp = Hp1 * Hp1 + Hp2;
        TensorScalarType Qp = (2.0 * Hp1 * Hp1 * Hp1 + 3.0 * Hp1 * Hp2 + 2.0 * Hp3) / 2.0;
        TensorScalarType Thetap = acos(Qp / pow(Pp, 3.0 / 2.0));

        TensorScalarType Ppp = 0.0;
        if(Hpp1 * Hpp1 + Hpp2 > 0.0)
            Ppp = Hpp1 * Hpp1 + Hpp2;
        TensorScalarType Qpp = (2.0 * Hpp1 * Hpp1 * Hpp1 + 3.0 * Hpp1 * Hpp2 + 2.0 * Hpp3) / 2.0;
        TensorScalarType Thetapp = acos(Qpp / pow(Ppp, 3.0 / 2.0));

        //Apply the analytic solutions for the SVD stress deviators
        TensorScalarType Sp1 = (Hp1 + 2.0 * sqrt(Hp1 * Hp1 + Hp2) * cos(Thetap / 3.0));
        TensorScalarType Sp2 = (Hp1 + 2.0 * sqrt(Hp1 * Hp1 + Hp2) * cos((Thetap + 4.0 * acos(-1.0)) / 3.0));
        TensorScalarType Sp3 = (Hp1 + 2.0 * sqrt(Hp1 * Hp1 + Hp2) * cos((Thetap + 2.0 * acos(-1.0)) / 3.0));

        TensorScalarType Spp1 = (Hpp1 + 2.0 * sqrt(Hpp1 * Hpp1 + Hpp2) * cos(Thetapp / 3.0));
        TensorScalarType Spp2 = (Hpp1 + 2.0 * sqrt(Hpp1 * Hpp1 + Hpp2) * cos((Thetapp + 4.0 * acos(-1.0)) / 3.0));
        TensorScalarType Spp3 = (Hpp1 + 2.0 * sqrt(Hpp1 * Hpp1 + Hpp2) * cos((Thetapp + 2.0 * acos(-1.0)) / 3.0));

        //Apply the Barlat yield function to get effective response.
        pnorm(cellOrdinal) = pow((pow(Sp1 - Spp1, mBarlatExponent) + pow(Sp1 - Spp2, mBarlatExponent)
                                  + pow(Sp1 - Spp3, mBarlatExponent) + pow(Sp2 - Spp1, mBarlatExponent)
                                  + pow(Sp2 - Spp2, mBarlatExponent) + pow(Sp2 - Spp3, mBarlatExponent)
                                  + pow(Sp3 - Spp1, mBarlatExponent) + pow(Sp3 - Spp2, mBarlatExponent)
                                  + pow(Sp3 - Spp3, mBarlatExponent))
                                 / 4.0,
                                 1.0 / mBarlatExponent);

        // normalize
        pnorm(cellOrdinal) /= mReferenceWeight;

        // pnorm
        pnorm(cellOrdinal) = pow(pnorm(cellOrdinal), p);
        pnorm(cellOrdinal) *= cellVolume(cellOrdinal);
    }
};
// class BarlatNormFunctor

/******************************************************************************/
/*! von Mises p-norm functor.

 Given a voigt tensor, compute the von Mises p-norm.
 Assumes single point integration.
 */
/******************************************************************************/
template<Plato::OrdinalType VoigtLength>
class VonMisesPNormFunctor
{
public:
    VonMisesPNormFunctor(Teuchos::ParameterList& params) :
            mScaleByVolume(true)
    {
        auto tNormalizeParams = params.sublist("Normalize");
        if (tNormalizeParams.isType<bool>("Volume Scaling"))
            mScaleByVolume = tNormalizeParams.get<bool>("Volume Scaling");
    }

    template<typename ResultScalarType, typename TensorScalarType, typename VolumeScalarType>
    KOKKOS_INLINE_FUNCTION void operator()(Plato::OrdinalType cellOrdinal,
                                       Plato::ScalarVectorT<ResultScalarType> pnorm,
                                       Plato::ScalarMultiVectorT<TensorScalarType> voigtTensor,
                                       Plato::OrdinalType p,
                                       Plato::ScalarVectorT<VolumeScalarType> cellVolume) const
    {
        pnorm(cellOrdinal) = 0.0;
        pnorm(cellOrdinal) += 0.5*pow(voigtTensor(cellOrdinal, 0) - voigtTensor(cellOrdinal, 1), 2.0);
        pnorm(cellOrdinal) += 0.5*pow(voigtTensor(cellOrdinal, 1) - voigtTensor(cellOrdinal, 2), 2.0);
        pnorm(cellOrdinal) += 0.5*pow(voigtTensor(cellOrdinal, 2) - voigtTensor(cellOrdinal, 0), 2.0);
        pnorm(cellOrdinal) += 3.0*pow(voigtTensor(cellOrdinal, 3), 2.0);
        pnorm(cellOrdinal) += 3.0*pow(voigtTensor(cellOrdinal, 4), 2.0);
        pnorm(cellOrdinal) += 3.0*pow(voigtTensor(cellOrdinal, 5), 2.0);

        pnorm(cellOrdinal) = pow(pnorm(cellOrdinal), p / 2.0);

        if (mScaleByVolume)
            pnorm(cellOrdinal) *= cellVolume(cellOrdinal);
    }

private:
    bool mScaleByVolume;
};
// class VonMisesPNormFunctor

/******************************************************************************/
/*! Abstract base class for computing a tensor norm.
 */
/******************************************************************************/
template<Plato::OrdinalType VoigtLength, typename EvalT>
class TensorNormBase
{

protected:
    Plato::Scalar mExponent;

public:
    TensorNormBase(Teuchos::ParameterList& params)
    {
        mExponent = params.get<Plato::Scalar>("Exponent");
    }

    virtual ~TensorNormBase()
    {
    }

    virtual void
    evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
             Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
             Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
             Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const = 0;

    virtual void postEvaluate(Plato::ScalarVector resultVector, Plato::Scalar resultScalar)
    {
        auto scale = pow(resultScalar, (1.0 - mExponent) / mExponent) / mExponent;
        auto numEntries = resultVector.size();
        Kokkos::parallel_for("scale vector", Kokkos::RangePolicy < Plato::OrdinalType > (0, numEntries),
                             KOKKOS_LAMBDA(Plato::OrdinalType entryOrdinal)
                             {
                                 resultVector(entryOrdinal) *= scale;
                             });
    }

    virtual void postEvaluate(Plato::Scalar& resultValue)
    {
        resultValue = pow(resultValue, 1.0 / mExponent);
    }

};
// class TensorNormBase

template<Plato::OrdinalType VoigtLength, typename EvalT>
class TensorPNorm : public TensorNormBase<VoigtLength, EvalT>
{

    TensorPNormFunctor<VoigtLength> mTensorPNorm;

public:

    TensorPNorm(Teuchos::ParameterList& params) :
            TensorNormBase<VoigtLength, EvalT>(params),
            mTensorPNorm(params)
    {
    }

    void evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
                  Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
                  Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
                  Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const
    {
        auto& tTensorPNorm = mTensorPNorm;
        Plato::OrdinalType numCells = result.extent(0);
        auto exponent = TensorNormBase<VoigtLength, EvalT>::mExponent;
        Kokkos::parallel_for("Compute PNorm", Kokkos::RangePolicy<Plato::OrdinalType>(0, numCells),
        KOKKOS_LAMBDA(Plato::OrdinalType cellOrdinal)
        {
            // compute tensor p-norm of tensor
            //
            tTensorPNorm(cellOrdinal, result, tensor, exponent, cellVolume);

        });
    }
};
// class TensorPNorm

template<Plato::OrdinalType VoigtLength, typename EvalT>
class BarlatNorm : public TensorNormBase<VoigtLength, EvalT>
{

    BarlatNormFunctor<VoigtLength> mBarlatNorm;

public:

    BarlatNorm(Teuchos::ParameterList& params) :
            TensorNormBase<VoigtLength, EvalT>(params),
            mBarlatNorm(params)
    {
    }

    void evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
                  Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
                  Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
                  Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const
    {
        Plato::OrdinalType numCells = result.extent(0);
        auto exponent = TensorNormBase<VoigtLength, EvalT>::mExponent;
        auto barlatNorm = mBarlatNorm;
        Kokkos::parallel_for("Compute Barlat Norm", Kokkos::RangePolicy<Plato::OrdinalType>(0, numCells),
                             KOKKOS_LAMBDA(Plato::OrdinalType cellOrdinal)
                             {
                                 // compute tensor p-norm of tensor
                                 //
                                 barlatNorm(cellOrdinal, result, tensor, exponent, cellVolume);

                             });
    }
};
// class BarlatNorm

template<Plato::OrdinalType VoigtLength, typename EvalT>
class WeightedNorm : public TensorNormBase<VoigtLength, EvalT>
{

    WeightedNormFunctor<VoigtLength> mWeightedNorm;

public:

    WeightedNorm(Teuchos::ParameterList& params) :
            TensorNormBase<VoigtLength, EvalT>(params),
            mWeightedNorm(params)
    {
    }

    void evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
                  Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
                  Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
                  Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const
    {
        Plato::OrdinalType numCells = result.extent(0);
        auto exponent = TensorNormBase<VoigtLength, EvalT>::mExponent;
        auto weightedNorm = mWeightedNorm;
        Kokkos::parallel_for("Compute Weighted Norm", Kokkos::RangePolicy<Plato::OrdinalType>(0, numCells),
                             KOKKOS_LAMBDA(Plato::OrdinalType cellOrdinal)
                             {
                                 // compute tensor p-norm of tensor
                                 //
                                 weightedNorm(cellOrdinal, result, tensor, exponent, cellVolume);

                             });
    }
};
// class WeightedNorm

template<Plato::OrdinalType VoigtLength, typename EvalT>
class VonMisesNorm : public TensorNormBase<VoigtLength, EvalT>
{
public:
    VonMisesNorm(Teuchos::ParameterList& params) :
            TensorNormBase<VoigtLength, EvalT>(params),
            mVonMisesPNorm(params)
    {
    }

    void evaluate(Plato::ScalarVectorT<typename EvalT::ResultScalarType> result,
                  Plato::ScalarMultiVectorT<typename EvalT::ResultScalarType> tensor,
                  Plato::ScalarMultiVectorT<typename EvalT::ControlScalarType> control,
                  Plato::ScalarVectorT<typename EvalT::ConfigScalarType> cellVolume) const
    {
        Plato::OrdinalType numCells = result.extent(0);
        auto exponent = TensorNormBase<VoigtLength, EvalT>::mExponent;
        auto vonMisesPNorm = mVonMisesPNorm;
        Kokkos::parallel_for("Compute Von Mises PNorm", Kokkos::RangePolicy<Plato::OrdinalType>(0, numCells),
                             KOKKOS_LAMBDA(Plato::OrdinalType cellOrdinal)
                             {
                                 // compute von mises p-norm of tensor
                                 //
                                 vonMisesPNorm(cellOrdinal, result, tensor, exponent, cellVolume);

                             });
    }

private:
    VonMisesPNormFunctor<VoigtLength> mVonMisesPNorm;
};
// class VonMisesNorm

template<Plato::OrdinalType VoigtLength, typename EvalT>
struct TensorNormFactory
{

    Teuchos::RCP<TensorNormBase<VoigtLength, EvalT>> create(Teuchos::ParameterList params)
    {

        Teuchos::RCP<TensorNormBase<VoigtLength, EvalT>> retval = Teuchos::null;

        if(params.isSublist("Normalize"))
        {
            auto normList = params.sublist("Normalize");
            auto normType = normList.get < std::string > ("Type");
            if(normType == "Barlat")
            {
                retval = Teuchos::rcp(new BarlatNorm<VoigtLength, EvalT>(params));
            }
            else if(normType == "Scalar")
            {
                retval = Teuchos::rcp(new WeightedNorm<VoigtLength, EvalT>(params));
            }
            else if(normType == "Von Mises")
            {
                retval = Teuchos::rcp(new VonMisesNorm<VoigtLength, EvalT>(params));
            }
            else
            {
                retval = Teuchos::rcp(new TensorPNorm<VoigtLength, EvalT>(params));
            }
        }
        else
        {
            retval = Teuchos::rcp(new TensorPNorm<VoigtLength, EvalT>(params));
        }

        return retval;
    }
};
// class TensorNormFactory

}// namespace Plato

#endif
