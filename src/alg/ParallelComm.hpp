/*
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
*/

#ifndef ANALYZE_PARALLEL_COMM_HPP
#define ANALYZE_PARALLEL_COMM_HPP

#include <Teuchos_RCPDecl.hpp>
#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_Comm.hpp>
#ifdef PLATO_EPETRA
#include <Epetra_SerialComm.h>
#include <Epetra_MpiComm.h>
#endif
#include <PlatoTypes.hpp>
#include <mpi.h>
#include <memory>

namespace Plato {
namespace Comm {

struct Machine {
  Teuchos::RCP<Teuchos::GlobalMPISession> mpiSession;
  Teuchos::RCP<const Teuchos::Comm<int>>  teuchosComm;
#ifdef PLATO_EPETRA
  std::shared_ptr<Epetra_Comm> epetraComm;
#endif

  Machine(MPI_Comm& localComm);

  Machine(int *argc, char ***argv);

  Machine();
};

unsigned size(Machine const& machine);
unsigned rank(Machine const& machine);

Plato::Scalar max(Machine const& machine, Plato::Scalar local);

Plato::Scalar min(Machine const& machine, Plato::Scalar local);

Plato::Scalar sum(Machine const& machine, Plato::Scalar local);

void allReduce(
    Machine const& machine, int n, const Plato::Scalar *local, Plato::Scalar *global);

}}  //end namespace Plato::comm

#endif
