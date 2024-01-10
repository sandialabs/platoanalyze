#pragma once

#include "Simp.hpp"

#include "SimplexFadTypes.hpp"
#include "AnalyzeMacros.hpp"

#include "ExpInstMacros.hpp"

namespace Plato
{
/**************************************************************************//**
* \brief Plane Stress J2 Plasticity Utilities Class
******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class J2PlasticityUtilities
{
  private:
    const Plato::Scalar mSqrt3Over2 = std::sqrt(3.0/2.0);
    const Plato::Scalar mSqrt2Over3 = std::sqrt(2.0/3.0);
    const Plato::Scalar mTwo = 2.0;

  public:
    /**************************************************************************//**
    * \brief Constructor
    ******************************************************************************/
    J2PlasticityUtilities()
    {
    }

    /******************************************************************************//**
     * \brief Update the plastic strain and backstress for a plastic step
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aPrevLocalState 2D container of previous local state variables
     * \param [in] aYieldSurfaceNormal 2D container of yield surface normal tensor components
     * \param [in] aHardeningModulusKinematic penalized kinematic hardening modulus
     * \param [out] aLocalState 2D container of local state variables to update
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION void
    updatePlasticStrainAndBackstressPlasticStep( 
                const Plato::OrdinalType       & aCellOrdinal,
                const Plato::ScalarMultiVector & aPrevLocalState,
                const Plato::ScalarMultiVector & aYieldSurfaceNormal,
                const Plato::Scalar            & aHardeningModulusKinematic,
                const Plato::ScalarMultiVector & aLocalState) const;

    /******************************************************************************//**
     * \brief Update the plastic strain and backstress for an elastic step
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aPrevLocalState 2D container of previous local state variables
     * \param [out] aLocalState 2D container of local state variables to update
    **********************************************************************************/
    KOKKOS_INLINE_FUNCTION void
    updatePlasticStrainAndBackstressElasticStep( 
                const Plato::OrdinalType       & aCellOrdinal,
                const Plato::ScalarMultiVector & aPrevLocalState,
                const Plato::ScalarMultiVector & aLocalState) const;

    /******************************************************************************//**
     * \brief Compute the yield surface normal and the norm of the deviatoric stress minus the backstress
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aDeviatoricStress deviatoric stress tensor
     * \param [in] aLocalState 2D container of local state variables to update
     * \param [out] aYieldSurfaceNormal 2D container of yield surface normal tensor components
     * \param [out] aDevStressMinusBackstressNorm norm(deviatoric_stress - backstress)
     *
     * Deviatoric stress \f$ \bf{s} \f$
     * Relative stress (\f$ \eta \f$)   = \f$ \bf{s} - \beta \f$
     * Yield surface normal (\f$ N \f$) = \f$ \frac{\eta}{\Vert \eta \Vert} \f$
     *
    **********************************************************************************/
    template<typename LocalStateT, typename StressT>
    KOKKOS_INLINE_FUNCTION void
    computeDeviatoricStressMinusBackstressNormalized(
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarMultiVectorT< StressT >         & aYieldSurfaceNormal,
                const Plato::ScalarVectorT< StressT >              & aDevStressMinusBackstressNorm) const;

    /******************************************************************************//**
     * \brief Compute the misfit between two distinct plastic strain tensors, i.e.
     *   aLocalStateOne - aLocalStateTwo.
     *
     * \tparam AViewType  input one POD type
     * \tparam BViewType  input two POD type
     * \tparam ResultType output POD type
     *
     * \param [in]  aCellOrdinal cell/element index
     * \param [in]  aLocalStateOne local states
     * \param [in]  aLocalStateTwo local states
     * \param [out] aOutput        output tensor
    **********************************************************************************/
    template<typename AViewType, typename BViewType, typename ResultType>
    KOKKOS_INLINE_FUNCTION void
    computePlasticStrainMisfit(const Plato::OrdinalType & aCellOrdinal,
                               const Plato::ScalarMultiVectorT< AViewType > & aLocalStateOne,
                               const Plato::ScalarMultiVectorT< BViewType > & aLocalStateTwo,
                               const Plato::ScalarMultiVectorT< ResultType > & aOutput) const;

    /******************************************************************************//**
     * \brief Fill the local residual vector with the plastic strain residual equation for plastic step
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aLocalState 2D container of local state variables
     * \param [in] aPrevLocalState 2D container of previous local state variables
     * \param [in] aYieldSurfaceNormal 2D container of yield surface normal tensor components
     * \param [out] aResult 2D container of local residual equations
    **********************************************************************************/
    template<typename LocalStateT, typename PrevLocalStateT, typename YieldSurfNormalT, typename ResultT>
    KOKKOS_INLINE_FUNCTION void
    fillPlasticStrainTensorResidualPlasticStep( 
                const Plato::OrdinalType                            & aCellOrdinal,
                const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
                const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
                const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const;

    /******************************************************************************//**
     * \brief Fill the local residual vector with the backstress residual equation for plastic step
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aHardeningModulusKinematic penalized kinematic hardening modulus
     * \param [in] aLocalState 2D container of local state variables
     * \param [in] aPrevLocalState 2D container of previous local state variables
     * \param [in] aYieldSurfaceNormal 2D container of yield surface normal tensor components
     * \param [out] aResult 2D container of local residual equations
    **********************************************************************************/
    template<typename ControlT, typename LocalStateT, typename PrevLocalStateT, 
             typename YieldSurfNormalT, typename ResultT>
    KOKKOS_INLINE_FUNCTION void
    fillBackstressTensorResidualPlasticStep( 
                const Plato::OrdinalType                            & aCellOrdinal,
                const ControlT                                      & aHardeningModulusKinematic,
                const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
                const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
                const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const;

    /******************************************************************************//**
     * \brief Fill the local residual vector with the plastic strain residual equation for elastic step
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aLocalState 2D container of local state variables
     * \param [in] aPrevLocalState 2D container of previous local state variables
     * \param [out] aResult 2D container of local residual equations
    **********************************************************************************/
    template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
    KOKKOS_INLINE_FUNCTION void
    fillPlasticStrainTensorResidualElasticStep( 
                const Plato::OrdinalType                            & aCellOrdinal,
                const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
                const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const;

    /******************************************************************************//**
     * \brief Fill the local residual vector with the backstress residual equation for plastic step
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aLocalState 2D container of local state variables
     * \param [in] aPrevLocalState 2D container of previous local state variables
     * \param [out] aResult 2D container of local residual equations
    **********************************************************************************/
    template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
    KOKKOS_INLINE_FUNCTION void
    fillBackstressTensorResidualElasticStep( 
                const Plato::OrdinalType                            & aCellOrdinal,
                const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
                const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
                const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const;

    /******************************************************************************//**
     * \brief Get the accumulated plastic strain from the current local state container
     * \param [in]  aLocalStates        2D container of local state variables
     * \param [out] aAccumPlasticStrain 2D container of accumulated plastic strains
    **********************************************************************************/
    template<typename LocalStateT>
    KOKKOS_INLINE_FUNCTION void
    getAccumulatedPlasticStrain(const Plato::OrdinalType & aCellOrdinal,
                                const Plato::ScalarMultiVectorT< LocalStateT > & aLocalStates,
                                const Plato::ScalarVectorT< LocalStateT > & aAccumPlasticStrain) const;

    /******************************************************************************//**
     * \brief Get the accumulated plastic strain from the current local state container
     * \param [in]  aLocalStates       2D container of local state variables
     * \param [out] aPlasticMultiplier 2D container of accumulated plastic strains
    **********************************************************************************/
    template<typename LocalStateT>
    KOKKOS_INLINE_FUNCTION void
    getPlasticMultiplierIncrement(const Plato::OrdinalType & aCellOrdinal,
                                  const Plato::ScalarMultiVectorT< LocalStateT > & aLocalStates,
                                  const Plato::ScalarVectorT< LocalStateT > & aPlasticMultiplier) const;

    /******************************************************************************//**
     * \brief Get the accumulated plastic strain from the current local state container
     * \param [in]  aLocalStates    2D container of local state variables
     * \param [out] aPlasticStrains 2D container of plastic strains
    **********************************************************************************/
    template<typename LocalStateT>
    KOKKOS_INLINE_FUNCTION void
    getPlasticStrainTensor(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT< LocalStateT > & aLocalStates,
                           const Plato::ScalarMultiVectorT< LocalStateT > & aPlasticStrains) const;

    /******************************************************************************//**
     * \brief Get the accumulated plastic strain from the current local state container
     * \param [in]  aLocalStates  2D container of local state variables
     * \param [out] aBackstresses 2D container of back-stresses
    **********************************************************************************/
    template<typename LocalStateT>
    KOKKOS_INLINE_FUNCTION void
    getBackstressTensor(const Plato::OrdinalType & aCellOrdinal,
                        const Plato::ScalarMultiVectorT< LocalStateT > & aLocalStates,
                        const Plato::ScalarMultiVectorT< LocalStateT > & aBackstresses) const;
};
// class J2PlasticityUtilities


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Update the plastic strain and backstress for a plastic step in 2D (Plane Strain)
  **********************************************************************************/
  template<>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::updatePlasticStrainAndBackstressPlasticStep( 
              const Plato::OrdinalType       & aCellOrdinal,
              const Plato::ScalarMultiVector & aPrevLocalState,
              const Plato::ScalarMultiVector & aYieldSurfaceNormal,
              const Plato::Scalar            & aHardeningModulusKinematic,
              const Plato::ScalarMultiVector & aLocalState) const
  {
    Plato::Scalar tMultiplier1 = aLocalState(aCellOrdinal, 1) * mSqrt3Over2;
    // Plastic Strain Tensor = {e_11, e_22, 2e_12, e_33}
    aLocalState(aCellOrdinal, 2) = aPrevLocalState(aCellOrdinal, 2) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 0);
    aLocalState(aCellOrdinal, 3) = aPrevLocalState(aCellOrdinal, 3) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 1);
    aLocalState(aCellOrdinal, 4) = aPrevLocalState(aCellOrdinal, 4) + mTwo * tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 2);
    aLocalState(aCellOrdinal, 5) = aPrevLocalState(aCellOrdinal, 5) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 3);

    Plato::Scalar tMultiplier2 = aLocalState(aCellOrdinal, 1) * mSqrt2Over3 * aHardeningModulusKinematic;
    // Backstress Tensor = {B_11, B_22, B_12, B_33}
    aLocalState(aCellOrdinal, 6) = aPrevLocalState(aCellOrdinal, 6) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 0);
    aLocalState(aCellOrdinal, 7) = aPrevLocalState(aCellOrdinal, 7) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 1);
    aLocalState(aCellOrdinal, 8) = aPrevLocalState(aCellOrdinal, 8) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 2);
    aLocalState(aCellOrdinal, 9) = aPrevLocalState(aCellOrdinal, 9) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 3);
  }

  /******************************************************************************//**
   * \brief Update the plastic strain and backstress for a plastic step in 3D
  **********************************************************************************/
  template<>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::updatePlasticStrainAndBackstressPlasticStep( 
              const Plato::OrdinalType       & aCellOrdinal,
              const Plato::ScalarMultiVector & aPrevLocalState,
              const Plato::ScalarMultiVector & aYieldSurfaceNormal,
              const Plato::Scalar            & aHardeningModulusKinematic,
              const Plato::ScalarMultiVector & aLocalState) const
  {
    Plato::Scalar tMultiplier1 = aLocalState(aCellOrdinal, 1) * mSqrt3Over2;
    // Plastic Strain Tensor
    aLocalState(aCellOrdinal, 2) = aPrevLocalState(aCellOrdinal, 2) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 0);
    aLocalState(aCellOrdinal, 3) = aPrevLocalState(aCellOrdinal, 3) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 1);
    aLocalState(aCellOrdinal, 4) = aPrevLocalState(aCellOrdinal, 4) + tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 2);
    aLocalState(aCellOrdinal, 5) = aPrevLocalState(aCellOrdinal, 5) + mTwo * tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 3);
    aLocalState(aCellOrdinal, 6) = aPrevLocalState(aCellOrdinal, 6) + mTwo * tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 4);
    aLocalState(aCellOrdinal, 7) = aPrevLocalState(aCellOrdinal, 7) + mTwo * tMultiplier1 * aYieldSurfaceNormal(aCellOrdinal, 5);

    Plato::Scalar tMultiplier2 = aLocalState(aCellOrdinal, 1) * mSqrt2Over3 * aHardeningModulusKinematic;
    // Backstress Tensor
    aLocalState(aCellOrdinal, 8) = aPrevLocalState(aCellOrdinal, 8) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 0);
    aLocalState(aCellOrdinal, 9) = aPrevLocalState(aCellOrdinal, 9) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 1);
    aLocalState(aCellOrdinal,10) = aPrevLocalState(aCellOrdinal,10) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 2);
    aLocalState(aCellOrdinal,11) = aPrevLocalState(aCellOrdinal,11) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 3);
    aLocalState(aCellOrdinal,12) = aPrevLocalState(aCellOrdinal,12) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 4);
    aLocalState(aCellOrdinal,13) = aPrevLocalState(aCellOrdinal,13) + tMultiplier2 * aYieldSurfaceNormal(aCellOrdinal, 5);
  }


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Update the plastic strain and backstress for an elastic step in 2D
  **********************************************************************************/
  template<>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::updatePlasticStrainAndBackstressElasticStep( 
              const Plato::OrdinalType       & aCellOrdinal,
              const Plato::ScalarMultiVector & aPrevLocalState,
              const Plato::ScalarMultiVector & aLocalState) const
  {
    // Plastic Strain Tensor
    aLocalState(aCellOrdinal, 2) = aPrevLocalState(aCellOrdinal, 2);
    aLocalState(aCellOrdinal, 3) = aPrevLocalState(aCellOrdinal, 3);
    aLocalState(aCellOrdinal, 4) = aPrevLocalState(aCellOrdinal, 4);
    aLocalState(aCellOrdinal, 5) = aPrevLocalState(aCellOrdinal, 5);

    // Backstress Tensor
    aLocalState(aCellOrdinal, 6) = aPrevLocalState(aCellOrdinal, 6);
    aLocalState(aCellOrdinal, 7) = aPrevLocalState(aCellOrdinal, 7);
    aLocalState(aCellOrdinal, 8) = aPrevLocalState(aCellOrdinal, 8);
    aLocalState(aCellOrdinal, 9) = aPrevLocalState(aCellOrdinal, 9);
  }

  /******************************************************************************//**
   * \brief Update the plastic strain and backstress for an elastic step in 3D
  **********************************************************************************/
  template<>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::updatePlasticStrainAndBackstressElasticStep( 
              const Plato::OrdinalType       & aCellOrdinal,
              const Plato::ScalarMultiVector & aPrevLocalState,
              const Plato::ScalarMultiVector & aLocalState) const
  {
    // Plastic Strain Tensor
    aLocalState(aCellOrdinal, 2) = aPrevLocalState(aCellOrdinal, 2);
    aLocalState(aCellOrdinal, 3) = aPrevLocalState(aCellOrdinal, 3);
    aLocalState(aCellOrdinal, 4) = aPrevLocalState(aCellOrdinal, 4);
    aLocalState(aCellOrdinal, 5) = aPrevLocalState(aCellOrdinal, 5);
    aLocalState(aCellOrdinal, 6) = aPrevLocalState(aCellOrdinal, 6);
    aLocalState(aCellOrdinal, 7) = aPrevLocalState(aCellOrdinal, 7);

    // Backstress Tensor
    aLocalState(aCellOrdinal, 8) = aPrevLocalState(aCellOrdinal, 8);
    aLocalState(aCellOrdinal, 9) = aPrevLocalState(aCellOrdinal, 9);
    aLocalState(aCellOrdinal,10) = aPrevLocalState(aCellOrdinal,10);
    aLocalState(aCellOrdinal,11) = aPrevLocalState(aCellOrdinal,11);
    aLocalState(aCellOrdinal,12) = aPrevLocalState(aCellOrdinal,12);
    aLocalState(aCellOrdinal,13) = aPrevLocalState(aCellOrdinal,13);
  }


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Compute the yield surface normal and the norm of the deviatoric stress
   * minus the backstress for 2D (Plane Strain)
   *
   * Deviatoric stress \f$ \bf{s} \f$
   * Relative stress (\f$ \eta \f$)   = \f$ \bf{s} - \beta \f$
   * Yield surface normal (\f$ N \f$) = \f$ \frac{\eta}{\Vert \eta \Vert} \f$
   *
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename StressT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::computeDeviatoricStressMinusBackstressNormalized(
              const Plato::OrdinalType                           & aCellOrdinal,
              const Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress,
              const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
              const Plato::ScalarMultiVectorT< StressT >         & aYieldSurfaceNormal,
              const Plato::ScalarVectorT< StressT >              & aDevStressMinusBackstressNorm) const
  {
    // Subtract the backstress from the deviatoric stress, i.e. compute relative stress
    aYieldSurfaceNormal(aCellOrdinal, 0) = aDeviatoricStress(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 6); // sigma_11
    aYieldSurfaceNormal(aCellOrdinal, 1) = aDeviatoricStress(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 7); // sigma_22
    aYieldSurfaceNormal(aCellOrdinal, 2) = aDeviatoricStress(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 8); // sigma_12
    aYieldSurfaceNormal(aCellOrdinal, 3) = aDeviatoricStress(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 9); // sigma_33

    // Compute the norm || stress_deviator - backstress ||
    aDevStressMinusBackstressNorm(aCellOrdinal) = sqrt(pow(aYieldSurfaceNormal(aCellOrdinal, 0), 2) +
                                                       pow(aYieldSurfaceNormal(aCellOrdinal, 1), 2) +
                                                       pow(aYieldSurfaceNormal(aCellOrdinal, 3), 2) +
                                                mTwo * pow(aYieldSurfaceNormal(aCellOrdinal, 2), 2));

    // Normalize the yield surface normal
    aYieldSurfaceNormal(aCellOrdinal, 0) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 1) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 2) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 3) /= aDevStressMinusBackstressNorm(aCellOrdinal);
  }

  /******************************************************************************//**
   * \brief Compute the yield surface normal and the norm of the deviatoric stress minus the backstress for 3D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename StressT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::computeDeviatoricStressMinusBackstressNormalized(
              const Plato::OrdinalType                           & aCellOrdinal,
              const Plato::ScalarMultiVectorT< StressT >         & aDeviatoricStress,
              const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
              const Plato::ScalarMultiVectorT< StressT >         & aYieldSurfaceNormal,
              const Plato::ScalarVectorT< StressT >              & aDevStressMinusBackstressNorm) const
  {
    // Subtract the backstress from the deviatoric stress
    aYieldSurfaceNormal(aCellOrdinal, 0) = aDeviatoricStress(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 8);
    aYieldSurfaceNormal(aCellOrdinal, 1) = aDeviatoricStress(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 9);
    aYieldSurfaceNormal(aCellOrdinal, 2) = aDeviatoricStress(aCellOrdinal, 2) - aLocalState(aCellOrdinal,10);
    aYieldSurfaceNormal(aCellOrdinal, 3) = aDeviatoricStress(aCellOrdinal, 3) - aLocalState(aCellOrdinal,11);
    aYieldSurfaceNormal(aCellOrdinal, 4) = aDeviatoricStress(aCellOrdinal, 4) - aLocalState(aCellOrdinal,12);
    aYieldSurfaceNormal(aCellOrdinal, 5) = aDeviatoricStress(aCellOrdinal, 5) - aLocalState(aCellOrdinal,13);

    // Compute the norm || stress_deviator - backstress ||
    aDevStressMinusBackstressNorm(aCellOrdinal) = sqrt(pow(aYieldSurfaceNormal(aCellOrdinal, 0), 2) +
                                                       pow(aYieldSurfaceNormal(aCellOrdinal, 1), 2) +
                                                       pow(aYieldSurfaceNormal(aCellOrdinal, 2), 2) +
                                                mTwo * pow(aYieldSurfaceNormal(aCellOrdinal, 3), 2) +
                                                mTwo * pow(aYieldSurfaceNormal(aCellOrdinal, 4), 2) +
                                                mTwo * pow(aYieldSurfaceNormal(aCellOrdinal, 5), 2));

    // Normalize the yield surface normal
    aYieldSurfaceNormal(aCellOrdinal, 0) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 1) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 2) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 3) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 4) /= aDevStressMinusBackstressNorm(aCellOrdinal);
    aYieldSurfaceNormal(aCellOrdinal, 5) /= aDevStressMinusBackstressNorm(aCellOrdinal);
  }


  /*******************************************************************************************/
  /*******************************************************************************************/


  /******************************************************************************//**
   * \brief Compute the misfit between two distinct plastic strain tensors, i.e.
   *   aLocalStateOne - aLocalStateTwo. Specialized for 3-D applications.
   *
   * \tparam AViewType  input one POD type
   * \tparam BViewType  input two POD type
   * \tparam ResultType output POD type
   *
   * \param [in]  aCellOrdinal   cell, i.e. element, index
   * \param [in]  aLocalStateOne local states
   * \param [in]  aLocalStateTwo local states
   * \param [out] aOutput        output tensor
  **********************************************************************************/
  template<>
  template<typename AViewType, typename BViewType, typename ResultType>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::computePlasticStrainMisfit(const Plato::OrdinalType & aCellOrdinal,
                                                       const Plato::ScalarMultiVectorT< AViewType > & aLocalStateOne,
                                                       const Plato::ScalarMultiVectorT< BViewType > & aLocalStateTwo,
                                                       const Plato::ScalarMultiVectorT< ResultType > & aOutput) const
  {
      aOutput(aCellOrdinal, 0) = aLocalStateOne(aCellOrdinal, 2) - aLocalStateTwo(aCellOrdinal, 2);
      aOutput(aCellOrdinal, 1) = aLocalStateOne(aCellOrdinal, 3) - aLocalStateTwo(aCellOrdinal, 3);
      aOutput(aCellOrdinal, 2) = aLocalStateOne(aCellOrdinal, 4) - aLocalStateTwo(aCellOrdinal, 4);
      aOutput(aCellOrdinal, 3) = aLocalStateOne(aCellOrdinal, 5) - aLocalStateTwo(aCellOrdinal, 5);
      aOutput(aCellOrdinal, 4) = aLocalStateOne(aCellOrdinal, 6) - aLocalStateTwo(aCellOrdinal, 6);
      aOutput(aCellOrdinal, 5) = aLocalStateOne(aCellOrdinal, 7) - aLocalStateTwo(aCellOrdinal, 7);
  }

  /******************************************************************************//**
   * \brief Compute the misfit between two distinct plastic strain tensors, i.e.
   *   aLocalStateOne - aLocalStateTwo. Specialized for 2-D applications. The plane
   *   strain assumption is used.
   *
   * \tparam AViewType  input one POD type
   * \tparam BViewType  input two POD type
   * \tparam ResultType output POD type
   *
   * \param [in]  aCellOrdinal   cell, i.e. element, index
   * \param [in]  aLocalStateOne local states
   * \param [in]  aLocalStateTwo local states
   * \param [out] aOutput        output tensor
  **********************************************************************************/
  template<>
  template<typename AViewType, typename BViewType, typename ResultType>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::computePlasticStrainMisfit(const Plato::OrdinalType & aCellOrdinal,
                                                       const Plato::ScalarMultiVectorT< AViewType > & aLocalStateOne,
                                                       const Plato::ScalarMultiVectorT< BViewType > & aLocalStateTwo,
                                                       const Plato::ScalarMultiVectorT< ResultType > & aOutput) const
  {
      aOutput(aCellOrdinal, 0) = aLocalStateOne(aCellOrdinal, 2) - aLocalStateTwo(aCellOrdinal, 2);
      aOutput(aCellOrdinal, 1) = aLocalStateOne(aCellOrdinal, 3) - aLocalStateTwo(aCellOrdinal, 3);
      aOutput(aCellOrdinal, 2) = aLocalStateOne(aCellOrdinal, 4) - aLocalStateTwo(aCellOrdinal, 4);
      aOutput(aCellOrdinal, 3) = aLocalStateOne(aCellOrdinal, 5) - aLocalStateTwo(aCellOrdinal, 5);
  }


  /*******************************************************************************************/
  /*******************************************************************************************/

  
  /******************************************************************************//**
   * \brief Fill the local residual vector with the plastic strain residual equation
   *   for plastic step in 2D (Plane Strain)
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename YieldSurfNormalT, typename ResultT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::fillPlasticStrainTensorResidualPlasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
              const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const
  {
      // epsilon^{p}_{11}
      aResult(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 2) - aPrevLocalState(aCellOrdinal, 2)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 0);

      // epsilon^{p}_{22}
      aResult(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 3) - aPrevLocalState(aCellOrdinal, 3)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 1);

      // epsilon^{p}_{12}
      aResult(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 4) - aPrevLocalState(aCellOrdinal, 4)
                             - mTwo * mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 2);

      // epsilon^{p}_{33}
      aResult(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 5) - aPrevLocalState(aCellOrdinal, 5)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 3);
  }

  /******************************************************************************//**
   * \brief Fill the local residual vector with the plastic strain residual equation for plastic step in 3D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename YieldSurfNormalT, typename ResultT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::fillPlasticStrainTensorResidualPlasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
              const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const
  {
    aResult(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 2) - aPrevLocalState(aCellOrdinal, 2)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 0);
    aResult(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 3) - aPrevLocalState(aCellOrdinal, 3)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 1);
    aResult(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 4) - aPrevLocalState(aCellOrdinal, 4)
                             - mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 2);
    aResult(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 5) - aPrevLocalState(aCellOrdinal, 5)
                             - mTwo * mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 3);
    aResult(aCellOrdinal, 6) = aLocalState(aCellOrdinal, 6) - aPrevLocalState(aCellOrdinal, 6)
                             - mTwo * mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 4);
    aResult(aCellOrdinal, 7) = aLocalState(aCellOrdinal, 7) - aPrevLocalState(aCellOrdinal, 7)
                             - mTwo * mSqrt3Over2 * aLocalState(aCellOrdinal, 1) * aYieldSurfaceNormal(aCellOrdinal, 5);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Fill the local residual vector with the backstress residual equation for plastic step in 2D
  **********************************************************************************/
  template<>
  template<typename ControlT, typename LocalStateT, typename PrevLocalStateT, 
           typename YieldSurfNormalT, typename ResultT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::fillBackstressTensorResidualPlasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const ControlT                                      & aHardeningModulusKinematic,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
              const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const
  {
      // backstress_{11}
      aResult(aCellOrdinal, 6) = aLocalState(aCellOrdinal, 6) - aPrevLocalState(aCellOrdinal, 6)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 0);

      // backstress_{22}
      aResult(aCellOrdinal, 7) = aLocalState(aCellOrdinal, 7) - aPrevLocalState(aCellOrdinal, 7)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 1);

      // backstress_{12}
      aResult(aCellOrdinal, 8) = aLocalState(aCellOrdinal, 8) - aPrevLocalState(aCellOrdinal, 8)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 2);

      // backstress_{33}
      aResult(aCellOrdinal, 9) = aLocalState(aCellOrdinal, 9) - aPrevLocalState(aCellOrdinal, 9)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1)
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 3);
  }

  /******************************************************************************//**
   * \brief Fill the local residual vector with the backstress residual equation for plastic step in 3D
  **********************************************************************************/
  template<>
  template<typename ControlT, typename LocalStateT, typename PrevLocalStateT, 
           typename YieldSurfNormalT, typename ResultT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::fillBackstressTensorResidualPlasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const ControlT                                      & aHardeningModulusKinematic,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< YieldSurfNormalT > & aYieldSurfaceNormal,
              const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const
  {
    aResult(aCellOrdinal, 8) = aLocalState(aCellOrdinal, 8) - aPrevLocalState(aCellOrdinal, 8)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 0);
    aResult(aCellOrdinal, 9) = aLocalState(aCellOrdinal, 9) - aPrevLocalState(aCellOrdinal, 9)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 1);
    aResult(aCellOrdinal,10) = aLocalState(aCellOrdinal,10) - aPrevLocalState(aCellOrdinal,10)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 2);
    aResult(aCellOrdinal,11) = aLocalState(aCellOrdinal,11) - aPrevLocalState(aCellOrdinal,11)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 3);
    aResult(aCellOrdinal,12) = aLocalState(aCellOrdinal,12) - aPrevLocalState(aCellOrdinal,12)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 4);
    aResult(aCellOrdinal,13) = aLocalState(aCellOrdinal,13) - aPrevLocalState(aCellOrdinal,13)
                               - mSqrt2Over3 * aLocalState(aCellOrdinal, 1) 
                               * aHardeningModulusKinematic * aYieldSurfaceNormal(aCellOrdinal, 5);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Fill the local residual vector with the plastic strain residual equation
   * for elastic step in 2D (Plane Strain)
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::fillPlasticStrainTensorResidualElasticStep( 
              const Plato::OrdinalType                              & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >        & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >    & aPrevLocalState,
              const Plato::ScalarMultiVectorT< ResultT >            & aResult ) const
  {
    aResult(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 2) - aPrevLocalState(aCellOrdinal, 2); // epsilon_11
    aResult(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 3) - aPrevLocalState(aCellOrdinal, 3); // epsilon_22
    aResult(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 4) - aPrevLocalState(aCellOrdinal, 4); // epsilon_12
    aResult(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 5) - aPrevLocalState(aCellOrdinal, 5); // epsilon_33
  }

  /******************************************************************************//**
   * \brief Fill the local residual vector with the plastic strain residual equation for elastic step in 3D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::fillPlasticStrainTensorResidualElasticStep( 
              const Plato::OrdinalType                              & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >        & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >    & aPrevLocalState,
              const Plato::ScalarMultiVectorT< ResultT >            & aResult ) const
  {
    aResult(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 2) - aPrevLocalState(aCellOrdinal, 2);
    aResult(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 3) - aPrevLocalState(aCellOrdinal, 3);
    aResult(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 4) - aPrevLocalState(aCellOrdinal, 4);
    aResult(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 5) - aPrevLocalState(aCellOrdinal, 5);
    aResult(aCellOrdinal, 6) = aLocalState(aCellOrdinal, 6) - aPrevLocalState(aCellOrdinal, 6);
    aResult(aCellOrdinal, 7) = aLocalState(aCellOrdinal, 7) - aPrevLocalState(aCellOrdinal, 7);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Fill the local residual vector with the backstress residual equation for
   *   elastic step in 2D (Plane Strain)
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::fillBackstressTensorResidualElasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const
  {
    aResult(aCellOrdinal, 6) = aLocalState(aCellOrdinal, 6) - aPrevLocalState(aCellOrdinal, 6); // sigma_11
    aResult(aCellOrdinal, 7) = aLocalState(aCellOrdinal, 7) - aPrevLocalState(aCellOrdinal, 7); // sigma_22
    aResult(aCellOrdinal, 8) = aLocalState(aCellOrdinal, 8) - aPrevLocalState(aCellOrdinal, 8); // sigma_12
    aResult(aCellOrdinal, 9) = aLocalState(aCellOrdinal, 9) - aPrevLocalState(aCellOrdinal, 9); // sigma_33
  }

  /******************************************************************************//**
   * \brief Fill the local residual vector with the backstress residual equation for elastic step in 3D
  **********************************************************************************/
  template<>
  template<typename LocalStateT, typename PrevLocalStateT, typename ResultT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::fillBackstressTensorResidualElasticStep( 
              const Plato::OrdinalType                            & aCellOrdinal,
              const Plato::ScalarMultiVectorT< LocalStateT >      & aLocalState,
              const Plato::ScalarMultiVectorT< PrevLocalStateT >  & aPrevLocalState,
              const Plato::ScalarMultiVectorT< ResultT >          & aResult ) const
  {
    aResult(aCellOrdinal, 8) = aLocalState(aCellOrdinal, 8) - aPrevLocalState(aCellOrdinal, 8);
    aResult(aCellOrdinal, 9) = aLocalState(aCellOrdinal, 9) - aPrevLocalState(aCellOrdinal, 9);
    aResult(aCellOrdinal,10) = aLocalState(aCellOrdinal,10) - aPrevLocalState(aCellOrdinal,10);
    aResult(aCellOrdinal,11) = aLocalState(aCellOrdinal,11) - aPrevLocalState(aCellOrdinal,11);
    aResult(aCellOrdinal,12) = aLocalState(aCellOrdinal,12) - aPrevLocalState(aCellOrdinal,12);
    aResult(aCellOrdinal,13) = aLocalState(aCellOrdinal,13) - aPrevLocalState(aCellOrdinal,13);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Get accumulated plastic strain - 2D implementation
  **********************************************************************************/
  template<>
  template<typename LocalStateT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::getAccumulatedPlasticStrain
  (const Plato::OrdinalType & aCellOrdinal,
   const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
   const Plato::ScalarVectorT<LocalStateT> & aAccumPlasticStrain) const
  {
      aAccumPlasticStrain(aCellOrdinal) = aLocalState(aCellOrdinal, 0);
  }

  /******************************************************************************//**
   * \brief Get accumulated plastic strain - 3D implementation
  **********************************************************************************/
  template<>
  template<typename LocalStateT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::getAccumulatedPlasticStrain
  (const Plato::OrdinalType & aCellOrdinal,
   const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
   const Plato::ScalarVectorT<LocalStateT> & aAccumPlasticStrain) const
  {
      aAccumPlasticStrain(aCellOrdinal) = aLocalState(aCellOrdinal, 0);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Get plastic multiplier increment - 2D implementation
  **********************************************************************************/
  template<>
  template<typename LocalStateT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::getPlasticMultiplierIncrement
  (const Plato::OrdinalType & aCellOrdinal,
   const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
   const Plato::ScalarVectorT<LocalStateT> & aPlasticMultiplier) const
  {
      aPlasticMultiplier(aCellOrdinal) = aLocalState(aCellOrdinal, 1);
  }

  /******************************************************************************//**
   * \brief Get plastic multiplier increment - 3D implementation
  **********************************************************************************/
  template<>
  template<typename LocalStateT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::getPlasticMultiplierIncrement
  (const Plato::OrdinalType & aCellOrdinal,
   const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
   const Plato::ScalarVectorT<LocalStateT> & aPlasticMultiplier) const
  {
      aPlasticMultiplier(aCellOrdinal) = aLocalState(aCellOrdinal, 1);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Get plastic strain tensor - 2D implementation
  **********************************************************************************/
  template<>
  template<typename LocalStateT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::getPlasticStrainTensor
  (const Plato::OrdinalType & aCellOrdinal,
   const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
   const Plato::ScalarMultiVectorT<LocalStateT> & aPlasticStrains) const
  {
      aPlasticStrains(aCellOrdinal, 0) = aLocalState(aCellOrdinal, 2);
      aPlasticStrains(aCellOrdinal, 1) = aLocalState(aCellOrdinal, 3);
      aPlasticStrains(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 4);
      aPlasticStrains(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 5);
  }

  /******************************************************************************//**
   * \brief Get plastic strain tensor - 3D implementation
  **********************************************************************************/
  template<>
  template<typename LocalStateT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::getPlasticStrainTensor
  (const Plato::OrdinalType & aCellOrdinal,
   const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
   const Plato::ScalarMultiVectorT<LocalStateT> & aPlasticStrains) const
  {
      aPlasticStrains(aCellOrdinal, 0) = aLocalState(aCellOrdinal, 2);
      aPlasticStrains(aCellOrdinal, 1) = aLocalState(aCellOrdinal, 3);
      aPlasticStrains(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 4);
      aPlasticStrains(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 5);
      aPlasticStrains(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 6);
      aPlasticStrains(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 7);
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Get plastic strain tensor - 2D implementation
  **********************************************************************************/
  template<>
  template<typename LocalStateT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<2>::getBackstressTensor
  (const Plato::OrdinalType & aCellOrdinal,
   const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
   const Plato::ScalarMultiVectorT<LocalStateT> & aBackstresses) const
  {
      aBackstresses(aCellOrdinal, 0) = aLocalState(aCellOrdinal, 6);
      aBackstresses(aCellOrdinal, 1) = aLocalState(aCellOrdinal, 7);
      aBackstresses(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 8);
      aBackstresses(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 9);
  }

  /******************************************************************************//**
   * \brief Get plastic strain tensor - 3D implementation
  **********************************************************************************/
  template<>
  template<typename LocalStateT>
  KOKKOS_INLINE_FUNCTION void
  J2PlasticityUtilities<3>::getBackstressTensor
  (const Plato::OrdinalType & aCellOrdinal,
   const Plato::ScalarMultiVectorT<LocalStateT> & aLocalState,
   const Plato::ScalarMultiVectorT<LocalStateT> & aBackstresses) const
  {
      aBackstresses(aCellOrdinal, 0) = aLocalState(aCellOrdinal, 8);
      aBackstresses(aCellOrdinal, 1) = aLocalState(aCellOrdinal, 9);
      aBackstresses(aCellOrdinal, 2) = aLocalState(aCellOrdinal, 10);
      aBackstresses(aCellOrdinal, 3) = aLocalState(aCellOrdinal, 11);
      aBackstresses(aCellOrdinal, 4) = aLocalState(aCellOrdinal, 12);
      aBackstresses(aCellOrdinal, 5) = aLocalState(aCellOrdinal, 13);
  }

} // namespace Plato
