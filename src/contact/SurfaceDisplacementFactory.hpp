#pragma once

#include "contact/ContactPair.hpp"
#include "contact/AbstractSurfaceDisplacement.hpp"
#include "contact/SurfaceDisplacement.hpp"
#include "contact/ProjectedSurfaceDisplacement.hpp"

#include "PlatoStaticsTypes.hpp"
#include "PlatoMesh.hpp"

#include "Teuchos_RCP.hpp"

namespace Plato
{

namespace Contact
{

template<typename EvaluationType>
class SurfaceDisplacementFactory 
{
public:
    SurfaceDisplacementFactory(){}

    Teuchos::RCP<SurfaceDisplacement<EvaluationType>>
    createChildContribution
    (const ContactSurface & aSurface,
     const Plato::Scalar  & aScale = 1.0)
    {
        return Teuchos::rcp( new SurfaceDisplacement<EvaluationType>(aSurface.childFaceLocalNodes(), aScale) );
    }

    Teuchos::RCP<ProjectedSurfaceDisplacement<EvaluationType>>
    createParentContribution
    (const ContactSurface & aSurface,
     Plato::Mesh            aMesh,
     const Plato::Scalar  & aScale = 1.0)
    {
        return Teuchos::rcp( new 
            ProjectedSurfaceDisplacement<EvaluationType>
            (aSurface.parentElements(), aSurface.mappedChildNodeLocations(), aSurface.elementWiseChildMap(), aMesh, aScale) 
        );
    }
};

}

}
