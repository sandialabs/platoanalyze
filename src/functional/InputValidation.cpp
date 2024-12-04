#include "InputValidation.hpp"

#include <algorithm>

#include "ParameterListUtilities.hpp"

namespace plato::functional
{
namespace
{
}  // namespace

bool affirm_input_mesh_blocks_match_mesh(const Teuchos::ParameterList& aParameterList, const Plato::Mesh& aMesh)
{
    const auto tBlocksFromMesh = aMesh->GetElementBlockNames();
    const auto tBlocksFromInput = element_block_names(aParameterList);
    if (tBlocksFromInput.size() != tBlocksFromMesh.size())
    {
        return false;
    }

    return std::all_of(tBlocksFromInput.cbegin(), tBlocksFromInput.cend(),
                       [&tBlocksFromMesh](const std::string& aBlockFromInput)
                       {
                           return std::any_of(tBlocksFromMesh.cbegin(), tBlocksFromMesh.cend(),
                                              [&aBlockFromInput](const std::string& aBlockFromMesh)
                                              { return aBlockFromInput == aBlockFromMesh; });
                       });
}

auto error_messages(const Teuchos::ParameterList& aParameterList, const Plato::Mesh& aMesh) -> std::string
{
    if (!affirm_input_mesh_blocks_match_mesh(aParameterList, aMesh))
    {
        auto tErrorMessage = std::stringstream{};
        tErrorMessage
            << "Mismatch in the element blocks in mesh with name " << aMesh->FileName()
            << " and the input xml file in platoanalyze.\n"
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
