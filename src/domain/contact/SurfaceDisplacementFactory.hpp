#pragma once

#include "Teuchos_RCP.hpp"
#include "domain/contact/AbstractSurfaceDisplacement.hpp"
#include "domain/contact/ContactPair.hpp"
#include "domain/contact/ProjectedSurfaceDisplacement.hpp"
#include "domain/contact/SurfaceDisplacement.hpp"
#include "linear_algebra/PlatoStaticsTypes.hpp"
#include "mesh/PlatoMesh.hpp"

namespace Plato
{

namespace Contact
{

template <typename EvaluationType>
class SurfaceDisplacementFactory
{
   public:
    SurfaceDisplacementFactory() {}

    Teuchos::RCP<SurfaceDisplacement<EvaluationType>> createChildContribution(const ContactSurface& aSurface,
                                                                              const Plato::Scalar& aScale = 1.0) const
    {
        return Teuchos::rcp(new SurfaceDisplacement<EvaluationType>(aSurface.childFaceLocalNodes(), aScale));
    }

    Teuchos::RCP<ProjectedSurfaceDisplacement<EvaluationType>> createParentContribution(
        const ContactSurface& aSurface, Plato::Mesh aMesh, const Plato::Scalar& aScale = 1.0) const
    {
        return Teuchos::rcp(new ProjectedSurfaceDisplacement<EvaluationType>(
            aSurface.parentElements(), aSurface.mappedChildNodeLocations(), aSurface.elementWiseChildMap(), aMesh,
            aScale));
    }
};

}  // namespace Contact

}  // namespace Plato
