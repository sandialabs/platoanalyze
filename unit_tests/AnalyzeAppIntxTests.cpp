#include <Teuchos_UnitTestHarness.hpp>
#include "AnalyzeAppIntxTests.hpp"

#include <Analyze_App.hpp>

std::shared_ptr<Plato::MPMD_App> createApp(std::string inputFile, std::string appFile);
void objectiveFiniteDifferenceTest(std::shared_ptr<Plato::MPMD_App> aApp, Plato::Scalar& val1, Plato::Scalar& val2, Plato::Scalar tol);
void objectiveFiniteDifferenceTest(std::string inputFile, std::string appFile, Plato::Scalar& val1, Plato::Scalar& val2, Plato::Scalar tol);

TEUCHOS_UNIT_TEST( AnalyzeAppTests, Reinitialize )
{ 
  /*
   * Create an MPMD_App, test it, reinitialize, and test it.
   * 
   */
  std::string inputFile = "Displacement_input.xml";
  std::string appFile = "Displacement_appfile.xml";
  Plato::Scalar val1(0.0), val2(0.0), tol(1e-7);

  auto tApp = createApp(inputFile, appFile);
  objectiveFiniteDifferenceTest(tApp, val1, val2, tol);

  TEST_FLOATING_EQUALITY(val1, val2, tol);

  tApp->reinitialize();

  objectiveFiniteDifferenceTest(tApp, val1, val2, tol);

  TEST_FLOATING_EQUALITY(val1, val2, tol);
}

TEUCHOS_UNIT_TEST( AnalyzeAppTests, MultipleProblemDefinitions )
{ 
  /*
   * Two operations with different ProblemDefinitions on
   * one performer.
   */
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=MultipleProblemDefinitions_input_1.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "MultipleProblemDefinitions_appfile.xml", true);

  Plato::MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<Plato::Scalar> stdStateIn(localIDs.size(),1.0);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // create output data
  //
  FauxSharedField fauxStateOut(localIDs.size(),0.0);
  std::vector<Plato::Scalar> stdStateOne(localIDs.size());
  std::vector<Plato::Scalar> stdStateTwo(localIDs.size());

  // solve 1
  //
  app.compute("Compute Displacement Solution 1");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateOne);

  // solve 2
  //
  app.compute("Compute Displacement Solution 2");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateTwo);

  for(int i=0; i<localIDs.size(); i++)
  {
    if( fabs(stdStateOne[i]) > 1e-16 )
      TEST_FLOATING_EQUALITY(stdStateOne[i], -stdStateTwo[i], 1e-10);
  }
}

TEUCHOS_UNIT_TEST( AnalyzeAppTests, OperationParameter )
{ 
  /*
   * One operation with a Parameter.
   */
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=OperationParameter_input.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "OperationParameter_appfile.xml", true);

  Plato::MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<Plato::Scalar> stdStateIn(localIDs.size(),0.5);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // set parameter
  //
  FauxParameter fauxParamIn("Traction X", "Compute Displacement Solution", 1.0);
  app.importDataT("Traction X", fauxParamIn);

  // create output data
  //
  FauxSharedField fauxStateOut(localIDs.size(),0.0);
  std::vector<Plato::Scalar> stdStateOne(localIDs.size());
  std::vector<Plato::Scalar> stdStateTwo(localIDs.size());

  // solve 1
  //
  app.compute("Compute Displacement Solution");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateOne);

  // set parameter
  //
  std::vector<Plato::Scalar> param(1,-1.0);
  fauxParamIn.setData(param);
  app.importDataT("Traction X", fauxParamIn);

  // solve 2
  //
  app.compute("Compute Displacement Solution");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateTwo);

  for(int i=0; i<localIDs.size(); i++)
  {
    if( fabs(stdStateOne[i]) > 1e-16 )
      TEST_FLOATING_EQUALITY(stdStateOne[i], -stdStateTwo[i], 1e-12);
  }
}

TEUCHOS_UNIT_TEST( AnalyzeAppTests, CellForcing )
{ 
  /*
   * One operation with cell forcing
   */
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=CellForcing_input.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "CellForcing_appfile.xml", true);

  Plato::MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<Plato::Scalar> stdStateIn(localIDs.size(),1.0);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // create output data
  //
  FauxSharedField fauxStateOut(localIDs.size(),0.0);
  std::vector<Plato::Scalar> stdStateOne(localIDs.size());

  // solve
  //
  app.compute("Compute Displacement Solution");

  // export data
  //
  app.exportDataT("Solution X", fauxStateOut);
  fauxStateOut.getData(stdStateOne);

//  for(int i=0; i<localIDs.size(); i++)
//  {
//    if( fabs(stdStateOne[i]) > 1e-16 )
//      TEST_FLOATING_EQUALITY(stdStateOne[i], -stdStateTwo[i], 1e-12);
//  }
}

TEUCHOS_UNIT_TEST( AnalyzeAppTests, EffectiveEnergy )
{ 
  /*
   * One operation with cell forcing
   */
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=EffectiveEnergy_input.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "EffectiveEnergy_appfile.xml", true);

  Plato::MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<Plato::Scalar> stdStateIn(localIDs.size(),1.0);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // create output data
  //
  FauxSharedValue fauxStateOut(1,0.0);
  std::vector<Plato::Scalar> stdStateOne(1);

  // solve
  //
  app.compute("Compute Objective Value");

  // export data
  //
  app.exportDataT("Objective Value", fauxStateOut);
  fauxStateOut.getData(stdStateOne);

  TEST_FLOATING_EQUALITY(stdStateOne[0], 9555665.134903859, 1e-12);
}

TEUCHOS_UNIT_TEST( AnalyzeAppTests, InternalEnergyGradX )
{ 
  
  int argc = 2;
  char exeName[] = "exeName";
  char arg1[] = "--input-config=InternalEnergyGradX_input.xml";
  char* argv[2] = {exeName, arg1};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", "InternalEnergyGradX_appfile.xml", true);

  Plato::MPMD_App app(argc, argv, myComm);

  app.initialize();

  std::vector<int> localIDs;
  app.exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxStateIn(localIDs.size());
  std::vector<Plato::Scalar> stdStateIn(localIDs.size(),1.0);

  // import data
  //
  fauxStateIn.setData(stdStateIn);
  app.importDataT("Topology", fauxStateIn);

  // create output data
  //
  FauxSharedField fauxStateOut(localIDs.size(),0.0);
  std::vector<Plato::Scalar> stdStateOut(localIDs.size());

  // solve
  //
  app.compute("Compute ObjectiveX");

  // export data
  //
  app.exportDataT("GradientX X", fauxStateOut);
  fauxStateOut.getData(stdStateOut);

//  TEST_FLOATING_EQUALITY(stdStateOut[0], 17308575.3656760529, 1e-10);

  app.exportDataT("GradientX Y", fauxStateOut);
  fauxStateOut.getData(stdStateOut);

//  TEST_FLOATING_EQUALITY(stdStateOut[0], 17308575.3656760529, 1e-11);

  app.exportDataT("GradientX Z", fauxStateOut);
  fauxStateOut.getData(stdStateOut);

//  TEST_FLOATING_EQUALITY(stdStateOut[0], 17308575.3656760529, 1e-12);
}

#ifdef PLATO_PARABOLIC
TEUCHOS_UNIT_TEST( AnalyzeAppTests, InternalEnergyHeatEq )
{ 
  std::string inputFile = "InternalEnergyHeatEq_input.xml";
  std::string appFile = "InternalEnergyHeatEq_appfile.xml";
  Plato::Scalar val1(0.0), val2(0.0), tol(1e-5);
  objectiveFiniteDifferenceTest(inputFile, appFile, val1, val2, tol);
  TEST_FLOATING_EQUALITY(val1, val2, tol);
}
#endif

TEUCHOS_UNIT_TEST( AnalyzeAppTests, InternalElectroelasticEnergy )
{ 
  std::string inputFile = "InternalElectroelasticEnergy_input.xml";
  std::string appFile = "InternalElectroelasticEnergy_appfile.xml";
  Plato::Scalar val1(0.0), val2(0.0), tol(1e-5);
  objectiveFiniteDifferenceTest(inputFile, appFile, val1, val2, tol);
  TEST_FLOATING_EQUALITY(val1, val2, tol);
}

TEUCHOS_UNIT_TEST( AnalyzeAppTests, EMStressPNorm )
{ 
  std::string inputFile = "EMStressPNorm_input.xml";
  std::string appFile = "EMStressPNorm_appfile.xml";
  Plato::Scalar val1(0.0), val2(0.0), tol(1e-7);
  objectiveFiniteDifferenceTest(inputFile, appFile, val1, val2, tol);
  TEST_FLOATING_EQUALITY(val1, val2, tol);
}

TEUCHOS_UNIT_TEST( AnalyzeAppTests, ThermoelasticEnergy )
{ 
  std::string inputFile = "InternalThermoelasticEnergy_input.xml";
  std::string appFile = "InternalThermoelasticEnergy_appfile.xml";
  Plato::Scalar val1(0.0), val2(0.0), tol(1e-7);
  objectiveFiniteDifferenceTest(inputFile, appFile, val1, val2, tol);
  TEST_FLOATING_EQUALITY(val1, val2, tol);
}

TEUCHOS_UNIT_TEST( AnalyzeAppTests, Displacement )
{ 
  std::string inputFile = "Displacement_input.xml";
  std::string appFile = "Displacement_appfile.xml";
  Plato::Scalar val1(0.0), val2(0.0), tol(1e-7);
  objectiveFiniteDifferenceTest(inputFile, appFile, val1, val2, tol);
  TEST_FLOATING_EQUALITY(val1, val2, tol);
}

void objectiveFiniteDifferenceTest(std::string inputFile, std::string appFile, Plato::Scalar& val1, Plato::Scalar& val2, Plato::Scalar tol)
{
  auto tApp = createApp(inputFile, appFile);
  objectiveFiniteDifferenceTest(tApp, val1, val2, tol);
}

std::shared_ptr<Plato::MPMD_App> createApp(std::string inputFile, std::string appFile)
{
  int argc = 2;
  char exeName[] = "exeName";
  std::stringstream input;
  input << "--input-config=" << inputFile;
  char* arg1 = strdup(input.str().c_str());
  char* argv[3] = {exeName, arg1, NULL};

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);

  setenv("PLATO_APP_FILE", appFile.c_str(), true);

  auto tApp = std::make_shared<Plato::MPMD_App>(argc, argv, myComm);

  tApp->initialize();

  return tApp;
}

void objectiveFiniteDifferenceTest(std::shared_ptr<Plato::MPMD_App> aApp, Plato::Scalar& val1, Plato::Scalar& val2, Plato::Scalar tol)
{

  std::vector<int> localIDs;
  aApp->exportDataMap(Plato::data::layout_t::SCALAR_FIELD, localIDs);

  // create input data
  //
  FauxSharedField fauxControlIn(localIDs.size());
  std::vector<Plato::Scalar> stdControlIn(localIDs.size(),1.0);

  // import data
  //
  fauxControlIn.setData(stdControlIn);
  aApp->importDataT("Topology", fauxControlIn);

  // create output data
  //

  // solve
  //
  aApp->compute("Compute Objective");

  // export data
  //
  FauxSharedField fauxObjGradOut(localIDs.size(),0.0);
  aApp->exportDataT("Objective Gradient", fauxObjGradOut);

  std::vector<Plato::Scalar> stdObjGradOut(localIDs.size());
  fauxObjGradOut.getData(stdObjGradOut);

  FauxSharedValue fauxObjValOut(1,0.0);
  aApp->exportDataT("Objective Value", fauxObjValOut);

  std::vector<Plato::Scalar> stdObjValOne(1);
  fauxObjValOut.getData(stdObjValOne);

  Plato::Scalar mag = 0.0;
  for(int iVal=0; iVal<localIDs.size(); iVal++){
    mag += stdObjGradOut[iVal]*stdObjGradOut[iVal];
  }
  mag = sqrt(mag);
  
  Plato::Scalar alpha = tol/mag;
  Plato::Scalar dval = 0.0;
  for(int iVal=0; iVal<localIDs.size(); iVal++){
    Plato::Scalar dz = alpha * stdObjGradOut[iVal];
    stdControlIn[iVal] -= dz;
    dval -= stdObjGradOut[iVal]*dz;
  }

  fauxControlIn.setData(stdControlIn);
  aApp->importDataT("Topology", fauxControlIn);
  aApp->compute("Compute Objective");

  aApp->exportDataT("Objective Value", fauxObjValOut);
  std::vector<Plato::Scalar> stdObjValTwo(1);
  fauxObjValOut.getData(stdObjValTwo);
  
  val1 = stdObjValOne[0]+dval;
  val2 = stdObjValTwo[0];
}
