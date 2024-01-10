//@HEADER
// ************************************************************************
//
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
//@HEADER

#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

//----------------------------------------------------------------------------
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_FileInputStream.hpp>
#include <Teuchos_ParameterEntry.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Time.hpp>
#include <PlatoTypes.hpp>

#include "alg/ParseInput.hpp"

namespace Plato {

void add_timings(
    Teuchos::ParameterList &problem,
    Teuchos::Time &         time_main,
    Comm::Machine const &   machine) {
  Teuchos::ParameterList &runtime = problem.sublist("Runtime");
  const std::string       time_doc =
      "Number of seconds of execution time on processor 0.";
  const Plato::Scalar sec = time_main.stop();
  if (Comm::rank(machine) == 0) {
    std::cout << "\nTotal Plato Execution Time: " << sec << " seconds\n";
  }
  runtime.set("Execution Time in Sec", sec, time_doc);
  runtime.get<Plato::Scalar>("Execution Time in Sec");
}

bool unused(
    const Teuchos::ParameterList &problem,
    const std::string &           path,
    std::string &                 os) {
  bool        all_unused = true;
  std::string not_used;
  for (Teuchos::ParameterList::ConstIterator i = problem.begin();
       i != problem.end(); ++i) {
    const Teuchos::ParameterEntry &entry_i = problem.entry(i);
    const std::string &            name_i = problem.name(i);
    if (entry_i.isList()) {
      const Teuchos::ParameterList &sublist = problem.sublist(name_i);
      const std::string             new_path(path + name_i + "::");
      const bool sublist_unused = unused(sublist, new_path, os);
      if (sublist_unused)
        not_used += "Block     " + path + name_i + '\n';
      else
        all_unused = false;
    } else if (entry_i.isUsed())
      all_unused = false;
    else {
      const std::string value(Teuchos::toString(entry_i.getAny(false)));
      not_used += "Parameter " + path + name_i + " : " + value + '\n';
    }
  }
  if (!all_unused) os += not_used;
  return all_unused;
}

// NVR 8-30-17: clang indicates that Function 'setParameters' is not needed and will not be emitted.  Can we delete this?
Teuchos::ParameterList setParameters(
    Teuchos::ParameterList &dest, const Teuchos::ParameterList &source) {
  for (Teuchos::ParameterList::ConstIterator i = source.begin();
       i != source.end(); ++i) {
    const std::string &            name_i = source.name(i);
    const Teuchos::ParameterEntry &entry_i = source.entry(i);
    if (entry_i.isList()) {
      Teuchos::ParameterList &dest_sublist =
          dest.sublist(name_i, false, entry_i.docString());
      const Teuchos::ParameterList &src_sublist =
          entry_i.getValue<Teuchos::ParameterList>(nullptr);
      setParameters(dest_sublist, src_sublist);
    } else {
      dest.setEntry(name_i, entry_i);
    }
  }
  return dest;
}

Teuchos::ParameterList
add_input_file(
    const Teuchos::ParameterList & problem,
    const Comm::Machine          & machine
)
{
  auto & runtime = problem.sublist("Runtime");
  auto  filename = runtime.get<std::string>("Input Config");

  Teuchos::ParameterList file_input(problem);

  if (Comm::rank(machine) == 0)
  {
    std::cout << " Reading input File: " << filename << std::endl;
  }

  auto Comm = machine.teuchosComm;
  Teuchos::updateParametersFromXmlFileAndBroadcast(filename, Teuchos::Ptr<Teuchos::ParameterList>(&file_input), *Comm);
  return file_input;
}

Teuchos::ParameterList input_file_parsing(
    int argc, char **argv, Comm::Machine const &machine)
{
  const bool throwExceptions = false;
  const bool recogniseAllOptions = true;
  const bool addOutputSetupOptions = false;

  Teuchos::CommandLineProcessor My_CLP(
      throwExceptions, recogniseAllOptions, addOutputSetupOptions);
  My_CLP.setDocString(
      "Plato will be used to demonstrate\n"
      "the application of Kokkos to shock dynamics.\n");
  const bool        query_def = false;
  bool              query = query_def;
  const std::string query_doc = "Query and Print machine parameters and exit.";
  My_CLP.setOption("query", "no-query", &query, query_doc.c_str());

  std::string       input_mesh = "input.exo";
  const std::string input_mesh_doc = "Name of the input exodus file to use.";
  My_CLP.setOption("input-mesh", &input_mesh, input_mesh.c_str());

  std::string       output_viz = "output";
  const std::string output_viz_doc = "Base name of visualization output.";
  My_CLP.setOption("output-viz", &output_viz, output_viz_doc.c_str());

  std::string input_config = "input.xml";
#ifdef OMEGA_H_USE_YAML
  const std::string input_config_doc =
      "Name of the input XML or YAML file to use.";
#else
  const std::string input_config_doc = "Name of the input XML file to use.";
#endif
  My_CLP.setOption("input-config", &input_config, input_config_doc.c_str());

  Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn =
      Teuchos::CommandLineProcessor::PARSE_ERROR;
  parseReturn = My_CLP.parse(argc, argv);
  if (parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    std::cerr << "Commnad line processing failed !" << std::endl;
    exit(-1);
  }

  Teuchos::ParameterList problem("Problem");
  if (parseReturn == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    problem.set("Query", query, query_doc);
    problem.set("Output Viz", output_viz, output_viz_doc);
    problem.set("Input Mesh", input_mesh, input_mesh_doc);

    Teuchos::ParameterList &runtime = problem.sublist(
        "Runtime", false,
        "System type parameters: Number of CPU run, platform name, code "
        "timers.\n"
        "To be printed at the end of the run for informational purposes.");
    runtime.set("Input Config", input_config, input_config_doc);
  } else {
    problem = Teuchos::ParameterList();
  }

  problem = add_input_file(problem, machine);

  return problem;
}

}  // namespace Plato
