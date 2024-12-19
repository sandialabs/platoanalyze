#include "InputValidation.hpp"

#include <algorithm>

#include "ParameterListUtilities.hpp"

namespace plato::functional
{
bool affirm_mesh_blocks_match_input(const Teuchos::ParameterList& aParameterList, const Plato::Mesh& aMesh)
{
    const auto tBlocksFromMesh = aMesh->GetElementBlockNames();
    const auto tBlocksFromInput = element_block_names(aParameterList);

    return std::all_of(tBlocksFromMesh.cbegin(), tBlocksFromMesh.cend(),
                       [&tBlocksFromInput](const std::string& aBlockFromMesh)
                       {
                           return std::any_of(tBlocksFromInput.cbegin(), tBlocksFromInput.cend(),
                                              [&aBlockFromMesh](const std::string& aBlockFromInput)
                                              { return aBlockFromInput == aBlockFromMesh; });
                       });
}

auto error_messages(const Teuchos::ParameterList& aParameterList, const Plato::Mesh& aMesh) -> std::string
{
    if (!affirm_mesh_blocks_match_input(aParameterList, aMesh))
    {
        auto tErrorMessage = std::stringstream{};
        tErrorMessage
            << "An element block in the mesh with file name " << aMesh->FileName()
            << " has no corresponding entry in the xml input file in platoanalyze.\n"
            << "Ensure that the platoanalyze input xml file has Domain entries matching all element blocks in the mesh."
            << "\nElement blocks in mesh:\n";
        const auto tBlocksFromMesh = aMesh->GetElementBlockNames();
        std::copy(tBlocksFromMesh.cbegin(), tBlocksFromMesh.cend(),
                  std::ostream_iterator<std::string>{tErrorMessage, "\n"});
        tErrorMessage << "Element blocks in input:\n";
        const auto tBlocksFromInput = element_block_names(aParameterList);
        std::copy(tBlocksFromInput.cbegin(), tBlocksFromInput.cend(),
                  std::ostream_iterator<std::string>{tErrorMessage, "\n"});
        return tErrorMessage.str();
    }
    return std::string{};
}

}  // namespace plato::functional
