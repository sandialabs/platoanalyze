#pragma once

#include <Teuchos_ParameterList.hpp>

#include "PlatoMask.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato {

    using OrdinalList = Plato::ScalarVectorT<Plato::OrdinalType>;

    /******************************************************************************/
    /*!
      \brief class for sequence entries
    */
    template <typename ElementType>
    class SequenceStep
    /******************************************************************************/
    {
        std::string mName;
        std::shared_ptr<Plato::Mask<ElementType::mNumSpatialDims>> mMask;

      public:
        decltype(mMask) getMask() const {return mMask;}

        template<int mNumDofsPerNode>
        void
        constrainInactiveNodes(
            const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                  Plato::ScalarVector                  aVector
        ) const
        {
            auto tNodes = mMask->getInactiveNodes();

            if(aMatrix->isBlockMatrix())
            {
                Plato::applyBlockConstraints<mNumDofsPerNode>
                    (aMatrix, aVector, tNodes);
            }
            else
            {
                Plato::applyConstraints<mNumDofsPerNode>
                    (aMatrix, aVector, tNodes);
            }
        }

        /******************************************************************************//**
         * \brief Constructor
         * \param [in] aInputParams SequenceStep definition
        **********************************************************************************/
        SequenceStep(
                  Plato::SpatialModel    & aSpatialModel,
            const Teuchos::ParameterList & aInputParams,
                  std::string              aName
        ) :
            mName(aName)
        {
            Plato::MaskFactory<ElementType::mNumSpatialDims> tMaskFactory;
            mMask = tMaskFactory.create(aSpatialModel.Mesh, aInputParams);
        }
    };


    /******************************************************************************/
    /*!
      \brief class for sequence entries
    */
    template <typename ElementType>
    class Sequence
    /******************************************************************************/
    {
        std::vector<Plato::SequenceStep<ElementType>> mSteps;

      public:
        const decltype(mSteps) & getSteps() const { return mSteps; }

        int getNumSteps() const { return mSteps.size(); }

        Sequence(
                  Plato::SpatialModel    & aSpatialModel,
            const Teuchos::ParameterList & aInputParams
        )
        {
            if (aInputParams.isSublist("Sequence"))
            {
                auto tSequenceParams = aInputParams.sublist("Sequence");
                if (!tSequenceParams.isSublist("Steps"))
                {
                    ANALYZE_THROWERR("Parsing 'Sequence'. Required 'Steps' list not found");
                }

                auto tStepsParams = tSequenceParams.sublist("Steps");
                for(auto tIndex = tStepsParams.begin(); tIndex != tStepsParams.end(); ++tIndex)
                {
                    const auto & tEntry  = tStepsParams.entry(tIndex);
                    const auto & tMyName = tStepsParams.name(tIndex);

                    if (!tEntry.isList())
                    {
                        ANALYZE_THROWERR("Parameter in 'Steps' list not valid.  Expect lists only.");
                    }

                    Teuchos::ParameterList& tStepParams = tStepsParams.sublist(tMyName);
                    mSteps.push_back({aSpatialModel, tStepParams, tMyName});
                }
            }
        }
    };
} // namespace Plato
