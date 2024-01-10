#ifndef DOT_PRODUCT_HPP
#define DOT_PRODUCT_HPP

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Dot product functor.
  
    Given two 2D-Views, compute the scalar product. Assumes single point integration.
*/
/******************************************************************************/
template<Plato::OrdinalType NumElements>
class DotProduct
{
public:
    DotProduct(){}
    ~DotProduct(){}

    template<typename ProductScalarType, typename ViewScalarTypeOne, typename ViewScalarTypeTwo>
    KOKKOS_INLINE_FUNCTION void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ViewScalarTypeOne> & aInputOne,
                                       const Plato::ScalarMultiVectorT<ViewScalarTypeTwo> & aInputTwo,
                                       const Plato::ScalarVectorT<ProductScalarType> & aOutput) const
    {
        aOutput(aCellOrdinal) = 0.0;
        for(Plato::OrdinalType tIndex = 0; tIndex < NumElements; tIndex++)
        {
            aOutput(aCellOrdinal) += aInputOne(aCellOrdinal, tIndex) * aInputTwo(aCellOrdinal, tIndex);
        }
    }
};
// class DotProduct

} // namespace Plato

#endif /* DOT_PRODUCT_HPP */
